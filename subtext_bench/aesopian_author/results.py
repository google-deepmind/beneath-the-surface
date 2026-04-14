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

"""Classes for storing the results of The Aesopian Author game."""

import dataclasses
from typing import Any, Dict, List, Optional, Union
from subtext_bench.aesopian_author import spaces

dataclass = dataclasses.dataclass


@dataclass
class AgentInfo:
  """Represents additional information about an agent."""

  last_observation: Optional[
      Union[
          spaces.AuthorObservation,
          spaces.CriticObservation,
          spaces.InquisitorObservation,
      ]
  ] = None
  last_action: Optional[
      Union[spaces.AuthorialAction, spaces.InterpretorAction]
  ] = None
  last_prompt: Optional[List[Any]] = None


@dataclass
class AesopianAuthorMetrics:
  """Metrics for The Aesopian Author game."""

  # Author metrics
  time_to_first_strike: Optional[int] = None
  time_of_death: Optional[int] = None
  num_succeeded: int = 0
  num_conformed: int = 0
  num_contested: int = 0
  num_transgressing: int = 0

  # Inquisitor metrics
  num_missed_transgressions: int = 0
  num_contested_interpretations: int = 0
  num_strikes_administered: int = 0
  inquisitor_critic_agreement_rate: float = 0.0

  # Critic metrics
  num_digressions_from_state: int = 0

  # Shared metrics
  num_agreements: int = 0
  w_score: int = 0
  d_score: int = 0
  c_score: int = 0

  def to_dict(self):
    """Converts the dataclass to a dictionary."""
    return {
        "time_to_first_strike": self.time_to_first_strike,
        "time_of_death": self.time_of_death,
        "num_succeeded": self.num_succeeded,
        "num_conformed": self.num_conformed,
        "num_contested": self.num_contested,
        "num_transgressing": self.num_transgressing,
        "num_missed_transgressions": self.num_missed_transgressions,
        "num_contested_interpretations": self.num_contested_interpretations,
        "num_strikes_administered": self.num_strikes_administered,
        "inquisitor_critic_agreement_rate": (
            self.inquisitor_critic_agreement_rate
        ),
        "num_digressions_from_state": self.num_digressions_from_state,
        "w_score": self.w_score,
        "d_score": self.d_score,
        "c_score": self.c_score,
    }


@dataclass
class GameResults:
  """Represents the results of The Aesopian Author game."""

  player_scores: Dict[str, int]
  stories_nd_interpretations: List[Dict[str, any]]
  game_state: spaces.AesopianAuthorGameState
  metrics: AesopianAuthorMetrics
  setting_description: str
  author_name: str
  author_profile: str
  author_alignment: str
  control_author: Optional[str] = None
