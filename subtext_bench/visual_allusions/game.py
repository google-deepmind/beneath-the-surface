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

"""Defines the run_game function for the Visual Allusions game."""

from subtext_bench.visual_allusions import environment
from subtext_bench.visual_allusions import players
from subtext_bench.visual_allusions import results


def run_game(
    env: environment.VisualAllusionsEnv,
    player_name_to_player: dict[str, players.VisualAllusionsPlayer],
    seed: int = 42,
    logging_method: str = "none",
):
  """Runs a Visual Allusions game."""
  env.reset(seed=seed)

  while not env.done:
    if logging_method != "none":
      env.render()

    agent = env.agent_selection
    player = player_name_to_player[agent]

    observation, _, termination, truncation, _, _ = env.last()

    if termination or truncation:
      action = None
    else:
      action, prompt_parts = player(observation)
      if prompt_parts:
        env.infos[agent].last_prompt = prompt_parts

    env.step(action)

  if logging_method != "none":
    env.render()
  env.close()

  player_to_storytelling_rounds = {}
  player_to_clue_types = {}
  for agent, info in env.infos.items():
    player_to_storytelling_rounds[agent] = info.num_storytelling_rounds
    player_to_clue_types[agent] = info.clue_types

  game_results = results.GameResults(
      player_scores=env.game_state.player_scores,
      player_granular_scores=env.game_state.player_granular_scores,
      player_sparks=env.sparks,
      player_num_storytelling_rounds=player_to_storytelling_rounds,
      player_clue_types=player_to_clue_types,
  )
  return game_results, env.infos
