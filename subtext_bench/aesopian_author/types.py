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

"""Types for The Aesopian Author game environment."""

import dataclasses
import enum
from typing import Any, Dict, List, Optional
import pydantic


class AesopianAuthorVariant(enum.Enum):
  """Variants defined by the context shared between the author and critic."""

  BASE = "base"
  SHARED_CONTEXT = "shared_context_only"
  LITERARY_HISTORY = "literary_history_only"
  FULL_CONTEXT = "full"


class AgentRole(enum.Enum):
  """Roles agents can have in the game."""

  STORYTELLER = "Storyteller"
  INQUISITOR = "Inquisitor"
  CRITIC = "Critic"


class InterpretedTopic(enum.Enum):
  """Interpretations of the story."""

  BANNED_TOPIC = "Banned Topic"
  CELEB_TOPIC = "Celeberated Topic"
  NEITHER_TOPIC = "Neither Topic"


@dataclasses.dataclass
class StoryInterpretation:
  """Interpretations of the story."""

  interpretation: InterpretedTopic
  explanation: Optional[str] = None

  def __str__(self) -> str:
    return f"{self.interpretation}\n\nExplanation: {self.explanation}"

  def to_dict(self):
    """Converts the dataclass to a dictionary."""
    return {
        "interpretation": self.interpretation.value,
        "explanation": self.explanation,
    }


@dataclasses.dataclass
class DossierEntry:
  """A single entry in the Inquisitor's Dossier."""

  round: int
  outcome: str
  inquisitor_interpretation: InterpretedTopic

  def __str__(self) -> str:
    """Returns a nicely formatted string representation of the dossier entry."""
    return (
        f"Round {self.round}:\n"
        f"  My Interpretation: {self.inquisitor_interpretation}\n"
        # f"  Outcome: {self.outcome}\n"
        "----------------------------------------------------------------------"
    )


@dataclasses.dataclass
class GameSetting:
  """Represents the game setting."""

  setting_description: str
  m_ban: str
  m_celeb: str
  author_profile: Dict[str, str]
  shared_context_stories: Optional[List[str]] = None


@dataclasses.dataclass
class LiteraryHistoryEntry:
  """A single entry in the Critic's Literary History."""

  round: int
  story: str


class AuthorOutput(pydantic.BaseModel):
  thinking: str
  story: str


class InterpretorOutput(pydantic.BaseModel):
  thinking: str
  interpretation: InterpretedTopic
  explanation: str


@dataclasses.dataclass
class LLMPlayerArgs:
  """Arguments for configuring an LLM-based player."""

  model_name: str = "gemini/gemini-2.5-flash"
  temperature: float = 0.7
  max_output_tokens: int = 4096
  thinking_budget: int = 1024
  structured_outputs: bool = True
  trim_messages: bool = False
  max_prompt_tokens: int = 40960

  @classmethod
  def from_dict(cls, d: Dict[str, Any]):
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in field_names})
