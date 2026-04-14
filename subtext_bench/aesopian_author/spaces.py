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

"""Dataclasses for The Aesopian Author game environment."""

import dataclasses
from typing import Dict, List, Optional
from subtext_bench.aesopian_author import types


@dataclasses.dataclass
class AuthorialAction:
  """Represents an action that an author can take."""

  current_story: Optional[str] = None
  thinking_trace: Optional[str] = None
  # Optional variable for LLM based players to store prompts used to query them.
  prompt_parts: Optional[List[str]] = None


@dataclasses.dataclass
class InterpretorAction:
  """Represents an action that an interpreter (critic/inquisitor) can take."""

  interpretation: Optional[types.StoryInterpretation] = None
  thinking_trace: Optional[str] = None
  # Optional variable for LLM based players to store prompts used to query them.
  prompt_parts: Optional[List[str]] = None


@dataclasses.dataclass
class AuthorObservation:
  """Represents an observation for an author."""

  interpretations: Optional[
      Dict[types.AgentRole, types.StoryInterpretation]
  ] = None


@dataclasses.dataclass
class CriticObservation:
  """Represents an observation for a critic."""

  current_story: Optional[str] = None
  inquisitor_interpretation: Optional[types.StoryInterpretation] = None


@dataclasses.dataclass
class InquisitorObservation:
  """Represents an observation for an inquisitor."""

  current_story: Optional[str] = None
  critic_interpretation: Optional[types.StoryInterpretation] = None
  round_outcome: Optional[str] = None


class AesopianAuthorGameState:
  """Represents the state of The Aesopian Author game."""

  def __init__(
      self,
      players: Dict[types.AgentRole, str],
      game_setting: types.GameSetting,
      aesopian_author_variant: types.AesopianAuthorVariant = types.AesopianAuthorVariant.LITERARY_HISTORY,
      shared_context_stories: Optional[List[str]] = None,
      max_rounds: int = 10,
      k_strikes: int = 3,
  ):
    self.players = players
    self.game_setting = game_setting
    self.m_ban = game_setting.m_ban
    self.m_celeb = game_setting.m_celeb
    self.setting_description = game_setting.setting_description
    self.author_profile = game_setting.author_profile
    self.aesopian_author_variant = aesopian_author_variant
    self.shared_context_stories = shared_context_stories

    self.player_scores = {player_id: 0 for player_id in players.values()}
    self.dossier: List[types.DossierEntry] = []
    self.literary_history: List[types.LiteraryHistoryEntry] = []

    self.round_number = 1
    self.round_stage = "STORYTELLING"  # STORYTELLING, INTERPRETATION, SCORING

    self.current_story: Optional[str] = None
    self.inquisitor_interpretation: Optional[types.StoryInterpretation] = None
    self.critic_interpretation: Optional[types.StoryInterpretation] = None
    self.round_outcome: Optional[str] = None

    self.receiver_turn = types.AgentRole.INQUISITOR

    self.max_rounds = max_rounds
    self.k_strikes = k_strikes
    self.author_strike_count = 0

  def rotate_receiver_turn(self):
    """Rotates the receiver turn."""
    if self.receiver_turn == types.AgentRole.INQUISITOR:
      self.receiver_turn = types.AgentRole.CRITIC
    elif self.receiver_turn == types.AgentRole.CRITIC:
      self.receiver_turn = types.AgentRole.STORYTELLER
    elif self.receiver_turn == types.AgentRole.STORYTELLER:
      self.receiver_turn = types.AgentRole.INQUISITOR
    else:
      raise ValueError("Receiver turn is not set.")

  def is_game_over(self) -> bool:
    """Checks if the game is over."""
    if self.round_number >= self.max_rounds:
      return True

    if self.author_strike_count >= self.k_strikes:
      return True

    return False
