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

"""Runs a game in The Aesopian Author environment."""

from typing import Dict
from subtext_bench.aesopian_author import environment
from subtext_bench.aesopian_author import players as aesopian_author_players
from subtext_bench.aesopian_author import results


def run_game(
    env: environment.AesopianAuthorEnv,
    player_name_to_player: Dict[
        str, aesopian_author_players.AesopianAuthorPlayer
    ],
    seed: int = 42,
    logging_method: str = "none",
):
  """Runs The Aesopian Author game."""
  env.reset(seed=seed)

  while not env.done:
    if logging_method != "none":
      env.render()

    agent = env.agent_selection
    player = player_name_to_player[agent]

    observation = env.observe(agent)
    action = player(observation)

    env.step(action)

  if logging_method != "none":
    env.render()
  env.close()

  author_profile = env.game_state.game_setting.author_profile
  game_results = results.GameResults(
      player_scores=env.game_state.player_scores,
      stories_nd_interpretations=env.stories_nd_interpretations,
      game_state=env.game_state,
      metrics=env.metrics,
      setting_description=env.game_state.game_setting.setting_description,
      author_name=author_profile["name"],
      author_profile=author_profile["profile"],
      author_alignment=author_profile["state_aligned"],
  )
  return game_results, env.infos
