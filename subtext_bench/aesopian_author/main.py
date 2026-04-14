# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for running and aggregating The Aesopian Author experiments.

This module provides functions to run multiple Aesopian Author games based on
configurations, save individual run data, and aggregate the results
"""

import argparse
import json
import os
import pickle
import random
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from subtext_bench.aesopian_author import constants
from subtext_bench.aesopian_author import environment
from subtext_bench.aesopian_author import game
from subtext_bench.aesopian_author import players
from subtext_bench.aesopian_author import types
from subtext_bench.core import utils as core_utils


def aggregate_run_stats(
    results: List[Dict[str, Any]], results_dir: str
) -> None:
  """Aggregates statistics from multiple Aesopian Author game runs."""
  run_level_data = []
  for result in results:
    run_data = {
        "setting_description": result["setting_description"],
        "author_name": result["author_name"],
        "author_profile": result["author_profile"],
        "author_alignment": result["author_alignment"],
        "control_author": result["control_author"],
    }
    run_data.update(result["metrics"].to_dict())
    run_level_data.append(run_data)

  df = pd.DataFrame(run_level_data)
  with open(os.path.join(results_dir, "run_level_metrics.csv"), "w") as f:
    df.to_csv(f, index=False)

  def get_agg_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    agg_metrics = {}
    for col in data.columns:
      if pd.api.types.is_numeric_dtype(data[col]):
        agg_metrics[col] = {
            "values": data[col].tolist(),
            "mean": data[col].mean(),
            "std": data[col].std(),
            "median": data[col].median(),
            "max": data[col].max(),
            "min": data[col].min(),
        }
    return agg_metrics

  # Aggregate metrics for all runs
  aggregated_metrics = get_agg_metrics(df)
  with open(os.path.join(results_dir, "aggregated_metrics.json"), "w") as f:
    json.dump(aggregated_metrics, f, indent=2, default=default_serializer)

  # Aggregate metrics by author alignment
  agg_metrics_by_alignment = {}
  for alignment, group in df.groupby("author_alignment"):
    agg_metrics_by_alignment[str(alignment)] = get_agg_metrics(group)

  with open(
      os.path.join(results_dir, "aggregated_metrics_by_alignment.json"), "w"
  ) as f:
    json.dump(agg_metrics_by_alignment, f, indent=2, default=default_serializer)


def run_game_instance(
    run_name: str,
    run_seed: int,
    exp_config: Dict[str, Any],
    shared_stories: Optional[List[str]],
    results_dir: str,
):
  """Runs a single Aesopian Author game."""
  core_utils.set_seed(run_seed)

  settings_path = os.path.join(
      exp_config.get("settings_path", constants.SETTINGS_DATA_DIR),
      "settings.json",
  )
  max_rounds = exp_config.get("max_rounds", 10)
  k_strikes = exp_config.get("k_strikes", 3)
  logging_method = exp_config.get("logging", "html")
  aesopian_author_variant = types.AesopianAuthorVariant(
      exp_config.get("aesopian_author_variant", "literary_history_only")
  )
  control_author = exp_config.get("control_author")
  player_configs = exp_config.get("players")
  if player_configs is None:
    raise ValueError("No players found in experiment config.")

  if control_author:
    for p_config in player_configs:
      if p_config["role"] == "Storyteller":
        p_config["control_author"] = control_author

  player_role_to_id = {
      types.AgentRole(p_config["role"]): p_config["player_id"]
      for p_config in player_configs
  }

  log_path = os.path.join(results_dir, f"{run_name}.log")

  env = environment.AesopianAuthorEnv(
      players=player_role_to_id,
      settings_path=settings_path,
      max_rounds=max_rounds,
      k_strikes=k_strikes,
      aesopian_author_variant=aesopian_author_variant,
      shared_context_stories=shared_stories,
      seed=run_seed,
      log_path=log_path,
  )
  env.metadata["render_modes"] = (
      [logging_method] if isinstance(logging_method, str) else logging_method
  )

  env.reset(seed=run_seed)
  game_setting = env.game_state.game_setting

  player_map = {}
  global_llm_settings = exp_config.get("llm_settings", {})
  for p_config in exp_config["players"]:
    p_id = p_config["player_id"]
    p_type = p_config["type"]
    p_role = types.AgentRole(p_config["role"])
    player_llm_args = None
    if p_type == "llm":
      player_llm_settings = p_config.get("llm_settings", {})
      merged_settings = global_llm_settings.copy()
      merged_settings.update(player_llm_settings)
      player_llm_args = types.LLMPlayerArgs.from_dict(merged_settings)

    player_map[p_id] = players.initialize_player(
        player_type=p_type,
        player_role=p_role,
        setting_description=game_setting.setting_description,
        m_ban=game_setting.m_ban,
        m_celeb=game_setting.m_celeb,
        author_profile=game_setting.author_profile,
        llm_args=player_llm_args,
        aesopian_author_variant=aesopian_author_variant,
        shared_context_stories=shared_stories,
    )

  game_results, infos = game.run_game(env, player_map, run_seed, logging_method)

  return game_results, infos


def default_serializer(obj):
  """Default JSON serializer."""
  if isinstance(obj, (np.integer, np.floating, np.bool_)):
    return obj.item()
  elif isinstance(obj, np.ndarray):
    return obj.tolist()
  else:
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def run_experiments(
    config: Dict[str, Any],
    exp_name_filter: Optional[str],
    overwrite: bool = False,
):
  """Performs multiple experiment runs and aggregates results."""
  for name, exp_config in config.items():
    if exp_name_filter and name != exp_name_filter:
      continue

    print(f"===== Running Experiment: {name} =====")
    results_base_dir = exp_config.get("results_base_dir", constants.OUTPUT_DIR)
    results_dir = os.path.join(results_base_dir, name)
    os.makedirs(results_dir, exist_ok=True)

    use_shared_context = exp_config.get(
        "aesopian_author_variant", "shared_context_only"
    ) in [
        "shared_context_only",
        "full",
    ]

    if use_shared_context:
      sc_conf = exp_config.get("shared_context", {})
      shared_stories = core_utils.load_shared_context_data(
          sc_data_dir=sc_conf.get("data_dir", constants.TMS_DATA_DIR),
          max_story_len=sc_conf.get("max_story_len", 5000),
          num_stories=sc_conf.get("num_stories", 10),
          seed=exp_config.get("seed", 42),
      )
    else:
      shared_stories = None

    game_results_all = []
    num_runs = exp_config.get("num_runs", 1)
    base_seed = exp_config.get("seed", random.randint(0, 10000))
    for i in range(num_runs):
      run_seed = base_seed + i
      run_name = f"{name}_run_{i}"
      pickle_path = os.path.join(results_dir, f"{run_name}_full_results.pkl")

      if os.path.exists(pickle_path) and not overwrite:
        print(f"--- Loading cached Run {i+1}/{num_runs} for {name} ---")
        with open(pickle_path, "rb") as f:
          run_data = pickle.load(f)
      else:
        print(f"--- Starting Run {i + 1}/{num_runs} (Seed: {run_seed}) ---")
        game_res, infos = run_game_instance(
            run_name=run_name,
            run_seed=run_seed,
            exp_config=exp_config,
            shared_stories=shared_stories,
            results_dir=results_dir,
        )
        run_data = {
            "game_results": game_res,
            "infos": infos,
        }
        with open(pickle_path, "wb") as f:
          pickle.dump(run_data, f)
      # Save results
      run_result = run_data["game_results"].__dict__
      metrics_path = os.path.join(results_dir, f"{run_name}_metrics.json")
      metrics_data = {
          "setting_description": run_result["setting_description"],
          "author_name": run_result["author_name"],
          "author_profile": run_result["author_profile"],
          "author_alignment": run_result["author_alignment"],
          "control_author": run_result["control_author"],
          "metrics": run_result["metrics"].to_dict(),
      }
      with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

      stories_path = os.path.join(
          results_dir, f"{run_name}_stories_and_interpretations.json"
      )
      stories_data = {
          "setting_description": run_result["setting_description"],
          "author_name": run_result["author_name"],
          "author_profile": run_result["author_profile"],
          "author_alignment": run_result["author_alignment"],
          "control_author": run_result["control_author"],
          "stories_and_interpretations": run_result[
              "stories_nd_interpretations"
          ],
      }
      with open(stories_path, "w") as f:
        json.dump(stories_data, f, indent=2)

      game_results_all.append(run_result)

      print(f"--- Finished Run {i + 1}/{num_runs} for {name} ---")

    aggregate_run_stats(game_results_all, results_dir)

    print(f"===== Finished Experiment: {name} =====")
    print(f"Results saved to {results_dir}")


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Run Aesopian Author experiments based on a JSON configuration file."
      )
  )
  parser.add_argument(
      "-c",
      "--config_file",
      type=str,
      required=True,
      help="Path to the JSON experiment configuration file.",
  )
  parser.add_argument(
      "-e",
      "--experiment_name",
      type=str,
      default=None,
      help=(
          "If provided, run only this specific experiment from the config file."
      ),
  )
  parser.add_argument(
      "-o",
      "--overwrite",
      action="store_true",
      help="Overwrite existing cached run results if they exist.",
  )
  args = parser.parse_args()

  with open(args.config_file, "r") as f:
    config = json.load(f)
  run_experiments(config, args.experiment_name, args.overwrite)


if __name__ == "__main__":
  main()
