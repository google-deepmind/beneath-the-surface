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

"""Main script for running Visual Allusions experiments.

This script reads experiment configurations from a JSON file, runs the
specified Visual Allusions game experiments, saves results for each run, and
aggregates statistics across runs for each experiment.
"""

import argparse
import collections
import json
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from subtext_bench.core import utils as core_utils
from subtext_bench.visual_allusions import constants
from subtext_bench.visual_allusions import environment
from subtext_bench.visual_allusions import game
from subtext_bench.visual_allusions import players
from subtext_bench.visual_allusions import results
from subtext_bench.visual_allusions import sparks as sparks_lib
from subtext_bench.visual_allusions import types


def aggregate_run_stats(
    game_results_all: List[results.GameResults],
) -> Dict[str, Any]:
  """Aggregates statistics from multiple Visual Allusions game runs.

  Args:
      game_results_all: A list of GameResults objects from multiple runs.

  Returns:
      A dictionary containing aggregated statistics, including average scores,
      win rates, clue type scores, and spark coefficients.
  """
  if not game_results_all:
    return {"error": "No runs to aggregate."}

  player_scores_all = [gr.player_scores for gr in game_results_all]
  player_names = list(player_scores_all[0].keys())
  player2average_scores = collections.defaultdict(float)
  for player_scores in player_scores_all:
    for player, score in player_scores.items():
      player2average_scores[player] += score
  for player in player_names:
    player2average_scores[player] /= len(player_scores_all)

  player2win_rate = collections.defaultdict(float)
  for player_scores in player_scores_all:
    max_score = max(player_scores.values())
    winners = [p for p, s in player_scores.items() if s == max_score]
    for winner in winners:
      player2win_rate[winner] += 1.0 / len(winners)  # Handle ties
  for player in player_names:
    player2win_rate[player] /= len(player_scores_all)

  players_clue_type_scores = {
      p: collections.defaultdict(list) for p in player_names
  }
  for gr in game_results_all:
    for player in player_names:
      clue_types = gr.player_clue_types[player]
      if not clue_types:
        for ct in ["obvious", "obscure", "just-right"]:
          players_clue_type_scores[player][ct].append(0)
        continue
      clue_type_counts = collections.Counter(clue_types)
      for ct in ["obvious", "obscure", "just-right"]:
        players_clue_type_scores[player][ct].append(
            clue_type_counts.get(ct, 0) / len(clue_types)
        )

  players_clue_type_scores_with_stats = {p: {} for p in player_names}
  for player, clue_scores in players_clue_type_scores.items():
    for clue_type, scores in clue_scores.items():
      players_clue_type_scores_with_stats[player][clue_type] = {
          "mean": float(np.mean(scores)),
          "std": float(np.std(scores)),
          "all": [float(s) for s in scores],
      }

  # Aggregate sparks and storyteller counts for spark calculation
  total_storyteller_counts = collections.defaultdict(int)
  total_unsym_sparks = collections.defaultdict(int)
  for gr in game_results_all:
    for player, count in gr.player_num_storytelling_rounds.items():
      total_storyteller_counts[player] += count
    for pair, count in gr.player_sparks.items():
      total_unsym_sparks[pair] += count

  sparks_dict = {
      "storyteller_counts": total_storyteller_counts,
      "unsym_sparks": total_unsym_sparks,
  }

  spark_coefficients = {}
  spark_significance = {}
  for i in range(len(player_names)):
    for j in range(i + 1, len(player_names)):
      p1, p2 = player_names[i], player_names[j]
      pair_key = ":".join(tuple(sorted((p1, p2))))
      spark_coefficients[pair_key] = sparks_lib.get_spark_a_b(
          sparks_dict, p1, p2, player_names
      )
      (
          p1_st_sig,
          p2_st_sig,
          stat_sig_symbol,
          p_val_p1_st,
          p_val_p2_st,
          p1_best_guesser_for_p2,
          p2_best_guesser_for_p1,
      ) = sparks_lib.chi_squared_test_spark(sparks_dict, player_names, p1, p2)
      spark_significance[pair_key] = {
          "p1_storyteller_test_p_value": p_val_p1_st,
          "p2_storyteller_test_p_value": p_val_p2_st,
          "p1_storyteller_test_significant": p1_st_sig,
          "p2_storyteller_test_significant": p2_st_sig,
          "is_p1_best_guesser_for_p2": p1_best_guesser_for_p2,
          "is_p2_best_guesser_for_p1": p2_best_guesser_for_p1,
          "stat_sig_symbol": stat_sig_symbol,
      }

  spark_coefficients["average_spark"] = np.mean(
      list(spark_coefficients.values())
  )

  results_dict = {
      "player_scores_all": player_scores_all,
      "player_average_scores": dict(player2average_scores),
      "player_win_rate": dict(player2win_rate),
      "clue_type_scores": players_clue_type_scores_with_stats,
      "spark_coefficients": spark_coefficients,
      "spark_significance": spark_significance,
  }

  granular_scores_all = [
      gr.player_granular_scores
      for gr in game_results_all
      if gr.player_granular_scores
  ]
  if granular_scores_all:
    player2avg_granular = {
        p: collections.defaultdict(float) for p in player_names
    }
    for run_scores in granular_scores_all:
      for player, scores in run_scores.items():
        for score_type, score in scores.items():
          player2avg_granular[player][score_type] += score
    for player in player_names:
      for score_type in player2avg_granular[player]:
        player2avg_granular[player][score_type] /= len(granular_scores_all)
    results_dict["player_average_granular_scores"] = {
        p: dict(s) for p, s in player2avg_granular.items()
    }

  return results_dict


def run_game_instance(
    run_name: str,
    run_seed: int,
    exp_config: Dict[str, Any],
    shared_stories: Optional[List[str]],
    results_dir: str,
) -> Tuple[results.GameResults, Dict[str, Any]]:
  """Runs a single Visual Allusions game instance using game.run_game."""
  core_utils.set_seed(run_seed)

  card_dir = exp_config.get("card_data_dir", constants.CARDS_DATA_DIR)
  game_instr = exp_config.get(
      "game_instructions", constants.VISUAL_ALLUSIONS_RULES
  )
  winning_score = exp_config.get("winning_score", 30)
  logging_method = exp_config.get("logging", ["html"])[0]

  card_files = core_utils.load_cards_data(card_dir)
  global_llm_settings = exp_config.get("llm_settings", {})

  player_map = {}
  for p_config in exp_config["players"]:
    p_id = p_config["player_id"]
    p_type = p_config["type"]
    partner_id = p_config.get("partner_id")
    player_llm_args = None
    if p_type == "llm":
      player_llm_settings = p_config.get("llm_settings", {})
      merged_settings = global_llm_settings.copy()
      merged_settings.update(player_llm_settings)
      player_llm_args = types.LLMPlayerArgs.from_dict(merged_settings)

    player_map[p_id] = players.initialize_player(
        player_id=p_id,
        player_type=p_type,
        llm_args=player_llm_args,
        partner_player_id=partner_id,
        shared_context_stories=shared_stories if partner_id else None,
        card_data_dir=card_dir,
        game_instructions=game_instr,
    )

  log_path = os.path.join(results_dir, f"{run_name}.log")
  env = environment.VisualAllusionsEnv(
      card_image_paths=card_files,
      card_data_dir=card_dir,
      num_players=len(player_map),
      agent_names=list(player_map.keys()),
      winning_score=winning_score,
      log_path=log_path,
  )
  env.metadata["render_modes"] = (
      [logging_method] if isinstance(logging_method, str) else logging_method
  )

  game_results, infos = game.run_game(env, player_map, run_seed, logging_method)

  return game_results, infos


def run_experiments(
    config: Dict[str, Any],
    exp_name_filter: Optional[str],
    overwrite: bool,
):
  """Runs experiments based on the provided configuration."""
  for name, exp_config in config.items():
    if exp_name_filter and name != exp_name_filter:
      continue

    print(f"===== Running Experiment: {name} =====")
    results_dir = os.path.join(exp_config["results_base_dir"], name)
    os.makedirs(results_dir, exist_ok=True)

    shared_stories = None
    if exp_config.get("shared_context"):
      sc_conf = exp_config["shared_context"]
      shared_stories = core_utils.load_shared_context_data(
          sc_data_dir=sc_conf.get("data_dir", constants.TMS_DATA_DIR),
          max_story_len=sc_conf.get("max_story_len", 5000),
          num_stories=sc_conf.get("num_stories", 10),
          seed=exp_config.get("seed", 42),
      )

    game_results_all = []
    num_runs = exp_config.get("num_runs", 1)
    base_seed = exp_config.get("seed", random.randint(0, 10000))

    for i in range(num_runs):
      run_seed = base_seed + i
      run_name = f"run_{i}"
      pickle_path = os.path.join(results_dir, f"{run_name}.pkl")

      if os.path.exists(pickle_path) and not overwrite:
        print(f"--- Loading cached Run {i+1}/{num_runs} for {name} ---")
        with open(pickle_path, "rb") as f:
          run_data = pickle.load(f)
      else:
        print(
            f"--- Starting Run {i+1}/{num_runs} for {name} (Seed:"
            f" {run_seed}) ---"
        )
        game_res, infos = run_game_instance(
            run_name, run_seed, exp_config, shared_stories, results_dir
        )
        run_data = {
            "game_results": game_res,
            "agent_infos": infos,
        }
        with open(pickle_path, "wb") as f:
          pickle.dump(run_data, f)
        print(f"--- Finished Run {i + 1}/{num_runs} for {name} ---")

      game_results_all.append(run_data["game_results"])

    print(f"--- Aggregating results for {name} ---")
    aggregated_stats = aggregate_run_stats(game_results_all)
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
      json.dump(aggregated_stats, f, indent=2)

    print(f"===== Finished Experiment: {name} =====")
    print(f"Results saved to {results_dir}")


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Run Visual Allusions experiments"
          " based on a JSON configuration file."
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
