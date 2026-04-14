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

"""Dataclasses for the Visual Allusions game environment."""

import dataclasses
import enum
from typing import Optional, Tuple
from subtext_bench.visual_allusions import types

Enum = enum.Enum
dataclass = dataclasses.dataclass


class VisualAllusionsGameState:
  """Manages the current state of a Visual Allusions game."""

  def __init__(
      self,
      players: list[str],
      deck: types.VisualAllusionsDeck,
      winning_score: int = 30,
  ):
    if not 3 <= len(players) <= 6:
      raise ValueError("Visual Allusions supports 3 to 6 players.")

    self.players = players
    self.num_players = len(self.players)
    self.deck = deck
    self.winning_score = winning_score
    self.initial_hand_size = 7 if self.num_players == 3 else 6
    self.player_hands = [
        self.deck.deal_cards(self.initial_hand_size)
        for _ in range(self.num_players)
    ]
    self.player_scores = {player: 0 for player in self.players}
    self.last_round_scores = {player: 0 for player in self.players}
    self.player_granular_scores = {
        player: {
            "storytelling": 0,
            "guessing": 0,
            "distractor": 0,
            "bonus": 0,
        }
        for player in self.players
    }
    self.storyteller_index = 0

    self.round_number = 1
    self.round_stage_types = [
        "storytelling",
        "guesser_card_play",
        "guesser_vote",
        "observe_only",
    ]
    self.round_stage = "storytelling"

    # MODIFIED: Stores (card_path, owner_index) to simplify tracking
    self.cards_in_play: list[Tuple[str, int]] = []
    self.storyteller_card = None

    self.storyteller_clue = ""
    self.player_votes: dict[str, str] = {}  # {voter_index: voted_card_path}
    self.discard_pile: list[str] = []

  def rotate_storyteller(self):
    self.storyteller_index = (self.storyteller_index + 1) % self.num_players

  def reset_round_state(self):
    self.cards_in_play = []
    self.storyteller_clue = ""
    self.player_votes = {}

  def is_game_over(self) -> bool:
    # Game ends when any player reaches the winning score or the deck is empty
    return not self.deck or any(
        score >= self.winning_score for score in self.player_scores.values()
    )

  def display_state(self):
    print("\n--- Current Game State ---")
    print(f"Round: {self.round_number}, Stage: {self.round_stage}")
    print(f"Storyteller: Player {self.storyteller_index}")
    print(f"Deck size: {len(self.deck)}")
    print(f"Scores: {self.player_scores}")
    print("--------------------------")


@dataclass
class AgentAction:
  """Represents an action that an agent can take."""

  card_to_play: Optional[str] = None
  card_to_vote: Optional[str] = None
  clue: Optional[str] = None
  thinking_trace: Optional[str] = None

  def is_dead(self) -> bool:
    return (
        self.card_to_play is None
        and self.card_to_vote is None
        and self.clue is None
    )


@dataclass
class AgentObservation:
  """Represents an observation for an agent."""

  round_number: int
  current_role: Optional[types.AgentRole]
  hand: list[str]
  clue: Optional[str]
  played_card: Optional[str]
  voting_deck: Optional[list[str]]
  storyteller_player_id: Optional[str]
  game_scores: dict[str, int]
  last_round_scores: dict[str, int]
  personal_score: int
  votes: Optional[dict[str, str]] = None
  storyteller_card: Optional[str] = None
