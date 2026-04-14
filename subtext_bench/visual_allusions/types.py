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

"""Types for Visual Allusions game."""

import dataclasses
import enum
import random
from typing import Any, Dict, List, Optional
import pydantic

Enum = enum.Enum
dataclass = dataclasses.dataclass


class StorytellerOutput(pydantic.BaseModel):
  thinking: str
  storyteller_card: int
  clue: str
  story: Optional[str] = None


class StorytellerCardChoiceOutput(pydantic.BaseModel):
  thinking: str
  storyteller_card: int


class StorytellerClueOutput(pydantic.BaseModel):
  thinking: str
  clue: str


class CardPlayOutput(pydantic.BaseModel):
  thinking: str
  played_card: int


class VoteOutput(pydantic.BaseModel):
  thinking: str
  voted_card: int


@dataclass
class LLMPlayerArgs:
  """Arguments for configuring an LLM-based Visual Allusions player."""

  model_name: str = "gemini/gemini-2.5-flash"
  temperature: float = 0.7
  max_output_tokens: int = 4096
  thinking_budget: int = 1024
  num_players: int = 4
  structured_outputs: bool = True
  maintain_chat_history: bool = False
  trim_messages: bool = False
  max_prompt_tokens: int = 40960

  @classmethod
  def from_dict(cls, d: Dict[str, Any]):
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in field_names})


class VisualAllusionsDeck:
  """Represents a deck of Visual Allusions cards."""

  def __init__(self, card_image_paths: List[str]):
    self.all_cards = card_image_paths
    self.deck = self.all_cards[:]
    random.shuffle(self.deck)

  def deal_cards(self, num_cards: int) -> List[str]:
    if len(self.deck) < num_cards:
      # Instead of raising an error, just deal the rest of the deck.
      # The game-over condition will handle the empty deck.
      dealt_cards = self.deck[:]
      self.deck = []
      return dealt_cards
    dealt_cards = self.deck[:num_cards]
    self.deck = self.deck[num_cards:]
    return dealt_cards

  def __len__(self):
    return len(self.deck)


class AgentRole(Enum):
  STORYTELLER = "storyteller"
  CARDPLAYER = "card_player"
  VOTER = "voter"
  OBSERVER = "observer"
