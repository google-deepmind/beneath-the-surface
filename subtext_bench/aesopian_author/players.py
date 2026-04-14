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

"""Module containing various player implementations for The Aesopian Author game."""

import abc
import json
import os
import random
import re
from typing import Dict, List, Optional

from subtext_bench.aesopian_author import constants
from subtext_bench.aesopian_author import spaces
from subtext_bench.aesopian_author import types
from subtext_bench.core import llm
from subtext_bench.core import utils as core_utils

AuthorObservation = spaces.AuthorObservation
AuthorialAction = spaces.AuthorialAction
CriticObservation = spaces.CriticObservation
InterpretorAction = spaces.InterpretorAction
InquisitorObservation = spaces.InquisitorObservation

StoryInterpretation = types.StoryInterpretation
DossierEntry = types.DossierEntry
InterpretedTopic = types.InterpretedTopic
LiteraryHistoryEntry = types.LiteraryHistoryEntry
AgentRole = types.AgentRole
AesopianAuthorVariant = types.AesopianAuthorVariant
AuthorOutput = types.AuthorOutput
InterpretorOutput = types.InterpretorOutput
LLMPlayerArgs = types.LLMPlayerArgs


class AesopianAuthorPlayer(abc.ABC):
  """Abstract base class for an Aesopian Author player."""

  def __init__(
      self,
      setting_description: str,
      m_ban: str,
      m_celeb: str,
  ):
    self.setting_description = setting_description
    self.m_ban = m_ban
    self.m_celeb = m_celeb


class AesopianAuthorWriterPlayer(AesopianAuthorPlayer):
  """Abstract base class for an Aesopian Author writer player."""

  def __init__(
      self,
      setting_description: str,
      m_ban: str,
      m_celeb: str,
      author_profile: Dict[str, str],
      aesopian_author_variant: AesopianAuthorVariant = AesopianAuthorVariant.SHARED_CONTEXT,
      shared_context_stories: Optional[List[str]] = None,
  ):
    super().__init__(setting_description, m_ban, m_celeb)
    self.author_profile = author_profile
    self.aesopian_author_variant = aesopian_author_variant
    self.shared_context_stories = shared_context_stories

  @abc.abstractmethod
  def write_story(self, observation: AuthorObservation) -> AuthorialAction:
    pass

  @abc.abstractmethod
  def receive_interpretations(
      self, observation: AuthorObservation
  ) -> AuthorialAction:
    pass

  def get_null_action(self) -> AuthorialAction:
    return AuthorialAction()

  def __call__(self, observation: AuthorObservation) -> AuthorialAction:
    if observation.interpretations is None:
      action = self.write_story(observation)
    else:
      action = self.receive_interpretations(observation)
    return action


class RandomAuthorPlayer(AesopianAuthorWriterPlayer):
  """A random Aesopian Author writer player."""

  def write_story(self, observation: AuthorObservation) -> AuthorialAction:
    """Writes a story by randomly choosing between topics."""
    if random.random() < 0.5:
      story = f"This is a story about {self.m_ban}"
    else:
      story = f"This is a story about {self.m_celeb}"
    return AuthorialAction(current_story=story)

  def receive_interpretations(
      self, observation: AuthorObservation
  ) -> AuthorialAction:
    """Receives the interpretations and does nothing."""
    return self.get_null_action()


class AesopianAuthorCriticPlayer(AesopianAuthorPlayer):
  """Abstract base class for an Aesopian Author critic player."""

  def __init__(
      self,
      setting_description: str,
      m_ban: str,
      m_celeb: str,
      aesopian_author_variant: AesopianAuthorVariant = AesopianAuthorVariant.SHARED_CONTEXT,
      shared_context_stories: Optional[List[str]] = None,
  ):
    super().__init__(setting_description, m_ban, m_celeb)
    self.aesopian_author_variant = aesopian_author_variant
    self.shared_context_stories = shared_context_stories

  def get_null_action(self) -> InterpretorAction:
    return InterpretorAction()

  @abc.abstractmethod
  def interpret_story(
      self,
      observation: CriticObservation,
  ) -> InterpretorAction:
    pass

  def __call__(self, observation: CriticObservation) -> InterpretorAction:
    if observation.inquisitor_interpretation is None:
      action = self.interpret_story(observation)
    else:
      action = self.get_null_action()
    return action


class RandomCriticPlayer(AesopianAuthorCriticPlayer):
  """A random Aesopian Author critic player."""

  def interpret_story(
      self,
      observation: CriticObservation,
  ) -> InterpretorAction:
    """Interprets the story by randomly choosing between the topics."""
    if random.random() < 0.5:
      interpretation = InterpretedTopic.BANNED_TOPIC
    else:
      interpretation = InterpretedTopic.CELEB_TOPIC
    return InterpretorAction(
        interpretation=StoryInterpretation(
            interpretation=interpretation, explanation="I guessed randomly."
        )
    )


class AesopianAuthorInquisitorPlayer(AesopianAuthorPlayer):
  """Abstract base class for an Aesopian Author inquisitor player."""

  def __init__(
      self,
      setting_description: str,
      m_ban: str,
      m_celeb: str,
      author_profile: Dict[str, str],
      dossier: Optional[List[DossierEntry]] = None,
  ):
    super().__init__(setting_description, m_ban, m_celeb)
    self.author_profile = author_profile
    self.dossier = dossier if dossier is not None else []
    self.current_round = 1
    self.current_interpretation: Optional[StoryInterpretation] = None

  def get_null_action(self) -> InterpretorAction:
    return InterpretorAction()

  def update_dossier(
      self,
      round_outcome: str,
  ):
    """Updates the dossier with the current round's data."""
    if self.current_interpretation is None:
      raise ValueError("Current interpretation is None.")
    self.dossier.append(
        DossierEntry(
            round=self.current_round,
            outcome=round_outcome,
            inquisitor_interpretation=self.current_interpretation.interpretation
        )
    )

  @abc.abstractmethod
  def interpret_story(
      self,
      observation: InquisitorObservation,
  ) -> InterpretorAction:
    pass

  def __call__(self, observation: InquisitorObservation) -> InterpretorAction:

    if observation.critic_interpretation is None:
      action = self.interpret_story(observation)
      self.current_interpretation = action.interpretation
    else:
      self.update_dossier(observation.round_outcome)
      action = self.get_null_action()
      self.current_round += 1
    return action


class RandomInquisitorPlayer(AesopianAuthorInquisitorPlayer):
  """A random Aesopian Author inquisitor player."""

  def interpret_story(
      self,
      observation: InquisitorObservation,
  ) -> InterpretorAction:
    """Interprets the story by randomly choosing between the topics."""
    if random.random() < 0.5:
      interpretation = InterpretedTopic.BANNED_TOPIC
    else:
      interpretation = InterpretedTopic.CELEB_TOPIC
    return InterpretorAction(
        interpretation=StoryInterpretation(
            interpretation=interpretation, explanation="I guessed randomly."
        )
    )


class LLMAuthorPlayer(AesopianAuthorWriterPlayer):
  """An Aesopian Author writer that uses an LLM to make decisions."""

  def __init__(
      self,
      setting_description: str,
      m_ban: str,
      m_celeb: str,
      author_profile: Dict[str, str],
      aesopian_author_variant: AesopianAuthorVariant = AesopianAuthorVariant.LITERARY_HISTORY,
      shared_context_stories: Optional[List[str]] = None,
      model_name: str = "gemini/gemini-2.5-flash",
      temperature: float = 0.7,
      max_output_tokens: int = 8192,
      thinking_budget: int = -1,
      instructions_dir: str = constants.INSTRUCTIONS_DIR,
      control_author: Optional[str] = None,
      structured_outputs: bool = True,
      trim_messages: bool = False,
      max_prompt_tokens: int = 40960,
  ):
    super().__init__(
        setting_description,
        m_ban,
        m_celeb,
        author_profile,
        aesopian_author_variant,
        shared_context_stories,
    )
    self.model_name = model_name
    self.instructions_dir = instructions_dir
    self.control_author = control_author
    self.structured_outputs = (
        structured_outputs if "gemma" not in model_name else False
    )
    self.temperature = temperature
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.maintain_chat_history = aesopian_author_variant in [
        AesopianAuthorVariant.LITERARY_HISTORY,
        AesopianAuthorVariant.FULL_CONTEXT,
    ]
    self.trim_messages = trim_messages
    self.max_prompt_tokens = max_prompt_tokens
    self.chat_history = []

    self.initialize_llm_client()

  def initialize_llm_client(self):
    self.llm_client = llm.LLMClient(
        model=self.model_name,
        temperature=self.temperature,
        max_tokens=self.max_output_tokens,
        thinking_budget_tokens=self.thinking_budget,
        trim_messages=self.trim_messages,
        max_prompt_tokens=self.max_prompt_tokens,
    )

  def __getstate__(self):
    """Return state values to be pickled."""
    state = self.__dict__.copy()
    # Exclude the unpickleable llm_client
    del state["llm_client"]
    return state

  def __setstate__(self, state):
    """Restore state from the unpickled state values."""
    self.__dict__.update(state)
    # Re-initialize the llm_client
    self.initialize_llm_client()

  def _get_system_prompt(self):
    """Gets the system prompt for The Aesopian Author game."""
    if self.control_author in ["default", "contextual"]:
      system_prompt_file = f"{self.instructions_dir}/author_prompts/control.md"
    elif self.control_author == "shared_context":
      system_prompt_file = (
          f"{self.instructions_dir}/author_prompts/control_shared_context.md"
      )
    else:
      system_prompt_file = (
          f"{self.instructions_dir}/author_prompts/{self.aesopian_author_variant.value}.md"
      )
    if os.path.exists(system_prompt_file):
      system_prompt = core_utils.load_file(system_prompt_file)
      if self.control_author:
        format_kwargs = {
            "author_name": self.author_profile["name"],
            "setting_description": self.setting_description,
            "m_ban": self.m_ban,
            "m_celeb": self.m_celeb,
        }
        if self.control_author == "shared_context":
          format_kwargs["shared_context_stories"] = (
              "\n".join(self.shared_context_stories)
              if self.shared_context_stories
              else ""
          )
        system_prompt = system_prompt.format(**format_kwargs)
      else:
        system_prompt = system_prompt.format(
            author_name=self.author_profile["name"],
            author_profile=self.author_profile["profile"],
            setting_description=self.setting_description,
            m_ban=self.m_ban,
            m_celeb=self.m_celeb,
            shared_context_stories="\n".join(self.shared_context_stories)
            if self.shared_context_stories
            else "",
        )
      system_prompt += (
          "\n\n Think step-by-step before you craft your response and enclose"
          " your reasoning in <think> and </think> tags before providing your"
          " final answer in the specified format."
      )
      return system_prompt
    else:
      raise ValueError(
          f"System prompt file {system_prompt_file} does not exist."
      )

  def _prepare_prompt_for_writing(self):
    """Prepares the prompt for writing a story."""
    prompt_parts = [
        "Time to write a story! Please write a story that is at least 1000"
        " words long."
    ]
    if not self.structured_outputs:
      prompt_parts.append(
          "\n\nEnclose your story in <story> and </story> tags."
      )
    return [core_utils.wrap_text_parts("".join(prompt_parts))]

  def _prepare_prompt_for_reception(
      self,
      observation: AuthorObservation,
  ):
    """Prepares the prompt for receiving interpretations from other agents.

    Args:
      observation: The AuthorObservation containing the interpretations.

    Returns:
      A list of strings forming the prompt.
    """
    prompt_parts = []
    critic_interpretation = observation.interpretations[AgentRole.CRITIC]
    inquisitor_interpretation = observation.interpretations[
        AgentRole.INQUISITOR
    ]
    prompt_parts.append("This is what the critic had to say about your story: ")
    prompt_parts.append(
        f"\nHis interpretation: {critic_interpretation.interpretation}"
    )
    prompt_parts.append(
        f"\n The explanation he gave was: {critic_interpretation.explanation}"
    )

    prompt_parts.append(
        "\n\nThis is what the inquisitor had to say about your story:"
    )
    prompt_parts.append(
        f"\nHis interpretation: {inquisitor_interpretation.interpretation}"
    )
    prompt_parts.append(
        "\n The explanation he gave was:"
        f" {inquisitor_interpretation.explanation}"
    )
    prompt_parts.append(
        "\n\n No need to do anything at the moment. Just analyse the"
        " interpretations and your story and plan how to best proceed next time"
        " you right another story."
    )

    return [core_utils.wrap_text_parts("".join(prompt_parts))]

  def _parse_writing_response(self, response_text, response_thinking=None):
    """Parses the LLM's response when writing a story.

    Args:
      response_text: The response object from the LLM call.
      response_thinking: The thinking process from the LLM if it exists.

    Returns:
      A tuple containing the story and the thinking trace.
    """
    if self.structured_outputs:
      response_json = json.loads(response_text)
      return response_json["story"], response_json["thinking"]
    else:
      thinking_match = re.search(
          r"<think>(.*?)</think>", response_text, re.DOTALL
      )
      thinking = thinking_match.group(1).strip() if thinking_match else ""

      story_match = re.search(r"<story>(.*?)</story>", response_text, re.DOTALL)
      story = story_match.group(1).strip() if story_match else ""
      if not story:
        # The story is the text after </think>
        story = response_text.split("</think>")[-1].strip()

      if response_thinking:
        thinking = (
            f"Internal thinking: {response_thinking}\n\nResponse thinking:"
            f" {thinking}"
        )

      return story, thinking

  def write_story(self, observation: AuthorObservation) -> AuthorialAction:
    prompt_parts = self._prepare_prompt_for_writing()
    messages = core_utils.create_messages(
        user_message_parts=prompt_parts,
        system_prompt=self._get_system_prompt(),
        chat_history=self.chat_history,
    )
    response_text, response_thinking = self.llm_client.generate(
        messages=messages,
        response_format=AuthorOutput if self.structured_outputs else None,
    )

    if self.maintain_chat_history:
      self.chat_history = core_utils.update_history(
          assistant_response=response_text,
          user_message_parts=prompt_parts,
          chat_history=self.chat_history,
      )

    story, thinking_trace = self._parse_writing_response(
        response_text, response_thinking
    )

    prompt_parts_to_return = [self._get_system_prompt()]
    prompt_parts_to_return.extend(prompt_parts)

    return AuthorialAction(
        current_story=story,
        thinking_trace=thinking_trace,
        prompt_parts=prompt_parts_to_return,
    )

  def receive_interpretations(
      self, observation: AuthorObservation
  ) -> AuthorialAction:
    if self.control_author in ["default", "shared_context"]:
      return self.get_null_action()
    prompt_parts = self._prepare_prompt_for_reception(observation)

    messages = core_utils.create_messages(
        user_message_parts=prompt_parts,
        system_prompt=self._get_system_prompt(),
        chat_history=self.chat_history,
    )
    response_text, _ = self.llm_client.generate(
        messages=messages,
    )

    if self.maintain_chat_history:
      self.chat_history = core_utils.update_history(
          assistant_response=response_text,
          user_message_parts=prompt_parts,
          chat_history=self.chat_history,
      )

    thinking_trace = response_text

    return AuthorialAction(
        thinking_trace=thinking_trace,
        prompt_parts=prompt_parts,
    )


class LLMInquisitorPlayer(AesopianAuthorInquisitorPlayer):
  """An Aesopian Author Inquisitor that uses an LLM to make decisions."""

  def __init__(
      self,
      setting_description: str,
      m_ban: str,
      m_celeb: str,
      author_profile: Dict[str, str],
      dossier: Optional[List[DossierEntry]] = None,
      model_name: str = "gemini-2.5-flash",
      temperature: float = 0.7,
      max_output_tokens: int = 4096,
      thinking_budget: int = -1,
      instructions_dir: str = constants.INSTRUCTIONS_DIR,
      structured_outputs: bool = True,
      trim_messages: bool = False,
      max_prompt_tokens: int = 40960,
  ):
    super().__init__(
        setting_description, m_ban, m_celeb, author_profile, dossier
    )
    self.model_name = model_name
    self.instructions_dir = instructions_dir
    self.structured_outputs = (
        structured_outputs if "gemma" not in model_name else False
    )
    self.temperature = temperature
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.trim_messages = trim_messages
    self.max_prompt_tokens = max_prompt_tokens

    self.initialize_llm_client()

  def initialize_llm_client(self):
    self.llm_client = llm.LLMClient(
        model=self.model_name,
        temperature=self.temperature,
        max_tokens=self.max_output_tokens,
        thinking_budget_tokens=self.thinking_budget,
        trim_messages=self.trim_messages,
        max_prompt_tokens=self.max_prompt_tokens,
    )

  def __getstate__(self):
    """Return state values to be pickled."""
    state = self.__dict__.copy()
    # Exclude the unpickleable llm_client
    del state["llm_client"]
    return state

  def __setstate__(self, state):
    """Restore state from the unpickled state values."""
    self.__dict__.update(state)
    # Re-initialize the llm_client
    self.initialize_llm_client()

  def _get_system_prompt(self):
    """Gets the system prompt for The Aesopian Author game."""
    system_prompt_file = (
        f"{self.instructions_dir}/inquisitor_prompts/default.md"
    )
    if os.path.exists(system_prompt_file):
      system_prompt = core_utils.load_file(system_prompt_file)
      if self.dossier:
        dossier_content = "\n".join(
            [str(dossier_entry) for dossier_entry in self.dossier]
        )
      else:
        dossier_content = "Dossier is currently empty."
      system_prompt = system_prompt.format(
          setting_description=self.setting_description,
          m_ban=self.m_ban,
          m_celeb=self.m_celeb,
          author_name=self.author_profile["name"],
          author_profile=self.author_profile["profile"],
          dossier_content=dossier_content,
      )
      system_prompt += (
          "\n\n Think step-by-step before you craft your response and enclose"
          " your reasoning in <think> and </think> tags before providing your"
          " final answer in the specified format."
      )
      return system_prompt

  def _prepare_prompt_for_interpretation(
      self,
      observation: InquisitorObservation,
  ):
    """Prepares the prompt for interpreting a story."""
    prompt_parts = [
        "Here is the story you need to interpret:",
        f"\n\n---\n\n{observation.current_story}\n\n---\n\n",
        "Please provide your interpretation.",
    ]
    if not self.structured_outputs:
      prompt_parts.append(
          "\n\nRespond in the following format:\n<interpretation> [Banned"
          " Topic, Celeberated Topic, or Neither Topic]"
          " </interpretation>\n<explanation> [Your explanation] </explanation>"
      )
    return [core_utils.wrap_text_parts("".join(prompt_parts))]

  def _parse_interpretation_response(
      self, response_text, response_thinking=None
  ):
    """Parses the LLM's response for interpretation."""
    if self.structured_outputs:
      response_json = json.loads(response_text)
      response_json["interpretation"] = InterpretedTopic(
          response_json["interpretation"]
      )
      return (
          response_json["interpretation"],
          response_json["explanation"],
          response_json["thinking"],
      )
    else:
      thinking_match = re.search(
          r"<think>(.*?)</think>", response_text, re.DOTALL
      )
      thinking = thinking_match.group(1).strip() if thinking_match else ""

      interpretation_match = re.search(
          r"<interpretation>(.*?)</interpretation>", response_text, re.DOTALL
      )
      interpretation_str = (
          interpretation_match.group(1).strip() if interpretation_match else ""
      )
      try:
        interpretation = InterpretedTopic(interpretation_str)
      except ValueError:
        interpretation = None

      explanation_match = re.search(
          r"<explanation>(.*?)</explanation>", response_text, re.DOTALL
      )
      explanation = (
          explanation_match.group(1).strip() if explanation_match else ""
      )

      # Remove thinking from explanation if it's there
      if thinking:
        explanation = explanation.replace(
            f"<think>{thinking}</think>", ""
        ).strip()
      if response_thinking:
        thinking = (
            f"Internal thinking: {response_thinking}\n\nResponse"
            f" Thinking:{thinking}"
        )

      return interpretation, explanation, thinking

  def interpret_story(
      self,
      observation: InquisitorObservation,
  ) -> InterpretorAction:
    """Interprets the story using the LLM."""
    prompt_parts = self._prepare_prompt_for_interpretation(observation)

    messages = core_utils.create_messages(
        user_message_parts=prompt_parts,
        system_prompt=self._get_system_prompt(),
    )
    response_text, response_thinking = self.llm_client.generate(
        messages=messages,
        response_format=InterpretorOutput if self.structured_outputs else None,
    )

    interpretation, explanation, thinking_trace = (
        self._parse_interpretation_response(response_text, response_thinking)
    )

    prompt_parts_to_return = [self._get_system_prompt()]
    prompt_parts_to_return.extend(prompt_parts)

    return InterpretorAction(
        interpretation=StoryInterpretation(
            interpretation=interpretation, explanation=explanation
        ),
        thinking_trace=thinking_trace,
        prompt_parts=prompt_parts_to_return,
    )


class LLMCriticPlayer(AesopianAuthorCriticPlayer):
  """An Aesopian Author Critic player that uses an LLM to make decisions."""

  def __init__(
      self,
      setting_description: str,
      m_ban: str,
      m_celeb: str,
      aesopian_author_variant: AesopianAuthorVariant = AesopianAuthorVariant.LITERARY_HISTORY,
      shared_context_stories: Optional[List[str]] = None,
      model_name: str = "gemini-2.5-flash",
      temperature: float = 0.7,
      max_output_tokens: int = 4096,
      thinking_budget: int = -1,
      instructions_dir: str = constants.INSTRUCTIONS_DIR,
      structured_outputs: bool = True,
      trim_messages: bool = False,
      max_prompt_tokens: int = 40960,
  ):
    super().__init__(
        setting_description,
        m_ban,
        m_celeb,
        aesopian_author_variant,
        shared_context_stories,
    )
    self.model_name = model_name
    self.instructions_dir = instructions_dir
    self.structured_outputs = (
        structured_outputs if "gemma" not in model_name else False
    )
    self.temperature = temperature
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.maintain_chat_history = aesopian_author_variant in [
        AesopianAuthorVariant.LITERARY_HISTORY,
        AesopianAuthorVariant.FULL_CONTEXT,
    ]
    self.trim_messages = trim_messages
    self.max_prompt_tokens = max_prompt_tokens
    self.chat_history = []

    self.initialize_llm_client()

  def initialize_llm_client(self):
    self.llm_client = llm.LLMClient(
        model=self.model_name,
        temperature=self.temperature,
        max_tokens=self.max_output_tokens,
        thinking_budget_tokens=self.thinking_budget,
        trim_messages=self.trim_messages,
        max_prompt_tokens=self.max_prompt_tokens,
    )

  def __getstate__(self):
    """Return state values to be pickled."""
    state = self.__dict__.copy()
    # Exclude the unpickleable llm_client
    del state["llm_client"]
    return state

  def __setstate__(self, state):
    """Restore state from the unpickled state values."""
    self.__dict__.update(state)
    # Re-initialize the llm_client
    self.initialize_llm_client()

  def _get_system_prompt(self):
    """Gets the system prompt for The Aesopian Author game."""
    system_prompt_file = os.path.join(
        self.instructions_dir,
        "critic_prompts",
        f"{self.aesopian_author_variant.value}.md",
    )
    if os.path.exists(system_prompt_file):
      system_prompt = core_utils.load_file(system_prompt_file)
      system_prompt = system_prompt.format(
          setting_description=self.setting_description,
          m_ban=self.m_ban,
          m_celeb=self.m_celeb,
          shared_context_stories="\n".join(self.shared_context_stories)
          if self.shared_context_stories
          else "",
      )
      system_prompt += (
          "\n\n Think step-by-step before you craft your response and enclose"
          " your reasoning in <think> and </think> tags before providing your"
          " final answer in the specified format."
      )
      return system_prompt
    else:
      raise ValueError(
          f"System prompt file {system_prompt_file} does not exist."
      )

  def _prepare_prompt_for_interpretation(
      self,
      observation: CriticObservation,
  ):
    """Prepares the prompt for interpreting a story."""
    prompt_parts = [
        "Here is the story you need to interpret:",
        f"\n\n---\n\n{observation.current_story}\n\n---\n\n",
        "Please provide your interpretation.",
    ]
    if not self.structured_outputs:
      prompt_parts.append(
          "\n\nRespond in the following format:\n<interpretation> [Banned"
          " Topic, Celeberated Topic, or Neither Topic]"
          " </interpretation>\n<explanation> [Your explanation] </explanation>"
      )
    return [core_utils.wrap_text_parts("".join(prompt_parts))]

  def _parse_interpretation_response(
      self, response_text, response_thinking=None
  ):
    """Parses the LLM's response for interpretation."""
    if self.structured_outputs:
      response_json = json.loads(response_text)
      response_json["interpretation"] = InterpretedTopic(
          response_json["interpretation"]
      )
      return (
          response_json["interpretation"],
          response_json["explanation"],
          response_json["thinking"],
      )
    else:
      thinking_match = re.search(
          r"<think>(.*?)</think>", response_text, re.DOTALL
      )
      thinking = thinking_match.group(1).strip() if thinking_match else ""

      interpretation_match = re.search(
          r"<interpretation>(.*?)</interpretation>", response_text, re.DOTALL
      )
      interpretation_str = (
          interpretation_match.group(1).strip() if interpretation_match else ""
      )
      try:
        interpretation = InterpretedTopic(interpretation_str)
      except ValueError:
        interpretation = None

      explanation_match = re.search(
          r"<explanation>(.*?)</explanation>", response_text, re.DOTALL
      )
      explanation = (
          explanation_match.group(1).strip() if explanation_match else ""
      )

      # Remove thinking from explanation if it's there
      if thinking:
        explanation = explanation.replace(
            f"<think>{thinking}</think>", ""
        ).strip()
      if response_thinking:
        thinking = f"Internal thinking: {response_thinking}\n\n{thinking}"

      return interpretation, explanation, thinking

  def interpret_story(
      self,
      observation: CriticObservation,
  ) -> InterpretorAction:
    """Interprets the story using the LLM."""
    prompt_parts = self._prepare_prompt_for_interpretation(observation)
    messages = core_utils.create_messages(
        user_message_parts=prompt_parts,
        system_prompt=self._get_system_prompt(),
        chat_history=self.chat_history,
    )
    response_text, response_thinking = self.llm_client.generate(
        messages=messages,
        response_format=InterpretorOutput if self.structured_outputs else None,
    )

    if self.maintain_chat_history:
      self.chat_history = core_utils.update_history(
          assistant_response=response_text,
          user_message_parts=prompt_parts,
          chat_history=self.chat_history,
      )

    interpretation, explanation, thinking_trace = (
        self._parse_interpretation_response(response_text, response_thinking)
    )

    prompt_parts_to_return = [self._get_system_prompt()]
    prompt_parts_to_return.extend(prompt_parts)

    return InterpretorAction(
        interpretation=StoryInterpretation(
            interpretation=interpretation, explanation=explanation
        ),
        thinking_trace=thinking_trace,
        prompt_parts=prompt_parts_to_return,
    )


def initialize_player(
    player_type: str,
    player_role: AgentRole,
    setting_description: str,
    m_ban: str,
    m_celeb: str,
    author_profile: Dict[str, str],
    llm_args: Optional[LLMPlayerArgs] = None,
    aesopian_author_variant: AesopianAuthorVariant = AesopianAuthorVariant.LITERARY_HISTORY,
    shared_context_stories: Optional[List[str]] = None,
    control_author: Optional[str] = None,
) -> AesopianAuthorPlayer:
  """Initializes a player based on the provided configuration."""
  if player_type == "llm":
    llm_args = llm_args or LLMPlayerArgs()
    if player_role == AgentRole.STORYTELLER:
      return LLMAuthorPlayer(
          setting_description=setting_description,
          m_ban=m_ban,
          m_celeb=m_celeb,
          author_profile=author_profile,
          aesopian_author_variant=aesopian_author_variant,
          shared_context_stories=shared_context_stories,
          control_author=control_author,
          **llm_args.__dict__,
      )

    elif player_role == AgentRole.CRITIC:
      return LLMCriticPlayer(
          setting_description=setting_description,
          m_ban=m_ban,
          m_celeb=m_celeb,
          aesopian_author_variant=aesopian_author_variant,
          shared_context_stories=shared_context_stories,
          **llm_args.__dict__,
      )
    elif player_role == AgentRole.INQUISITOR:
      return LLMInquisitorPlayer(
          setting_description=setting_description,
          m_ban=m_ban,
          m_celeb=m_celeb,
          author_profile=author_profile,
          **llm_args.__dict__,
      )
    else:
      raise ValueError(f"Unknown LLM player role: {player_role}")
  elif player_type == "random":
    if player_role == AgentRole.STORYTELLER:
      return RandomAuthorPlayer(
          setting_description=setting_description,
          m_ban=m_ban,
          m_celeb=m_celeb,
          author_profile=author_profile,
          aesopian_author_variant=aesopian_author_variant,
          shared_context_stories=shared_context_stories,
      )
    elif player_role == AgentRole.CRITIC:
      return RandomCriticPlayer(
          setting_description=setting_description,
          m_ban=m_ban,
          m_celeb=m_celeb,
          aesopian_author_variant=aesopian_author_variant,
          shared_context_stories=shared_context_stories,
      )
    elif player_role == AgentRole.INQUISITOR:
      return RandomInquisitorPlayer(
          setting_description=setting_description,
          m_ban=m_ban,
          m_celeb=m_celeb,
          author_profile=author_profile,
      )
    else:
      raise ValueError(f"Unknown random player role: {player_role}")
  else:
    raise ValueError(f"Unknown player type: {player_type}")
