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

"""Classes for storing results of a Visual Allusions game."""

import dataclasses
from typing import Dict, List, Optional, Tuple
from subtext_bench.visual_allusions import spaces

dataclass = dataclasses.dataclass


@dataclass
class AgentInfo:
  """Represents additional information about an agent."""

  all_observations: List[spaces.AgentObservation] = dataclasses.field(
      default_factory=list
  )
  all_actions: List[spaces.AgentAction] = dataclasses.field(
      default_factory=list
  )
  last_observation: Optional[spaces.AgentObservation] = None
  last_action: Optional[spaces.AgentAction] = None
  previous_votes: Optional[Dict[str, str]] = None
  previous_cards_in_play: Optional[List[str]] = None
  clue_types: List[str] = dataclasses.field(default_factory=list)
  num_storytelling_rounds: int = 0
  last_prompt: Optional[List[str]] = None


@dataclass
class GameResults:
  """Represents the results of a Visual Allusions game."""

  player_scores: Dict[str, int]
  player_granular_scores: Dict[str, Dict[str, int]]
  player_sparks: Dict[Tuple[str, str], int]
  player_num_storytelling_rounds: Dict[str, int]
  player_clue_types: Dict[str, List[str]]
