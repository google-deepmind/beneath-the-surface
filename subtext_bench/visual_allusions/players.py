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

"""Players for the Visual Allusions game."""

import abc
import json
import os
import random
import re
from typing import Any, List, Optional, Tuple

from subtext_bench.core import llm
from subtext_bench.core import utils as core_utils
from subtext_bench.visual_allusions import constants
from subtext_bench.visual_allusions import spaces
from subtext_bench.visual_allusions import types


class VisualAllusionsPlayer(abc.ABC):
  """Abstract base class for a Visual Allusions player."""

  def __init__(self, player_id: str):
    """Initializes a base Player.

    Args:
        player_id (str): The unique identifier for the player.
    """
    self.player_id = player_id
    self.current_scores = 0
    self.players_scores = {}

  def _update_scores(self, new_scores: dict[str, int]):
    """Updates the player's current score."""
    self.players_scores = new_scores
    self.current_score = new_scores[self.player_id]

  @abc.abstractmethod
  def choose_card_and_provide_clue(self, hand: list[str]):
    """Abstract method for the storyteller to choose a card and provide a clue.

    Args:
        hand: The storyteller's hand of cards (list of file paths).

    Returns:
        tuple: A tuple containing the chosen card (file path) and the text clue.
    """
    pass

  @abc.abstractmethod
  def choose_card_to_play(
      self, hand: list[str], clue: str, storyteller_player_id: str
  ):
    """Abstract method for a guesser to choose a card to play.

    Args:
        hand: The player's hand of cards (list of file paths).
        clue: The storyteller's clue for the round.
        storyteller_player_id: The player id of the storyteller.

    Returns:
        str: The file path of the chosen card from the hand.
    """
    pass

  @abc.abstractmethod
  def vote(
      self,
      played_cards: list[str],
      clue: str,
      voters_card: str,
      storyteller_player_id: str,
  ):
    """Abstract method for a guesser to vote for the storyteller's card.

    Args:
        played_cards: A list of cards to vote from.
        clue: The storyteller's clue for the round.
        voters_card: The card of the player voting (so that they do not vote for
          their own card).
        storyteller_player_id: The player id of the storyteller.

    Returns:
        str: The file path of the card the player is voting for.
    """
    pass

  def just_observe(
      self, observation: spaces.AgentObservation
  ):  # pylint: disable=unused-argument
    """Player just observes the game."""
    self._update_scores(observation.game_scores)
    return spaces.AgentAction(
        card_to_play=None,
        card_to_vote=None,
        clue=None,
        thinking_trace=None,
    )

  def __call__(
      self, observation: spaces.AgentObservation
  ) -> Tuple[spaces.AgentAction, Optional[List[Any]]]:

    prompt_parts = None
    if observation.current_role == types.AgentRole.STORYTELLER:
      chosen_card, clue, thinking, prompt_parts = (
          self.choose_card_and_provide_clue(hand=observation.hand)
      )
      action = spaces.AgentAction(
          card_to_play=chosen_card, clue=clue, thinking_trace=thinking
      )

    elif observation.current_role == types.AgentRole.CARDPLAYER:
      chosen_card, thinking, prompt_parts = self.choose_card_to_play(
          hand=observation.hand,
          clue=observation.clue,
          storyteller_player_id=observation.storyteller_player_id,
      )
      action = spaces.AgentAction(
          card_to_play=chosen_card, thinking_trace=thinking
      )

    elif observation.current_role == types.AgentRole.VOTER:
      voted_card, thinking, prompt_parts = self.vote(
          played_cards=observation.voting_deck,
          clue=observation.clue,
          voters_card=observation.played_card,
          storyteller_player_id=observation.storyteller_player_id,
      )
      action = spaces.AgentAction(
          card_to_vote=voted_card, thinking_trace=thinking
      )

    elif observation.current_role == types.AgentRole.OBSERVER:
      action = self.just_observe(observation)

    else:
      raise ValueError(f"Unknown current_role: {observation.current_role}")

    return action, prompt_parts


class SimplePlayer(VisualAllusionsPlayer):
  """A simple Visual Allusions player that always picks the first card in its hand."""

  def update_history(self, scores: Any, votes: Any):
    print(
        f"SimplePlayer {self.player_id} updated history with scores"
        f" {scores}, votes {votes}"
    )

  def choose_card_and_provide_clue(self, hand: list[str]):
    # Storyteller just picks the first card and gives a generic clue
    chosen_card = hand[0]
    clue = "a picture"
    thinking = "I can't think"
    prompt_parts = ["I don't have a prompt"]
    return chosen_card, clue, thinking, prompt_parts

  def choose_card_to_play(
      self, hand: list[str], clue: str, storyteller_player_id: str
  ):
    # Guesser just picks the first card in their hand
    chosen_card = hand[0]
    thinking = "I can't think"
    prompt_parts = ["I don't have a prompt"]
    return chosen_card, thinking, prompt_parts

  def vote(
      self,
      played_cards: list[str],
      clue: str,
      voters_card: str,
      storyteller_player_id: str,
  ):
    # Guesser just votes for the first card in played_cards
    voted_card = (
        played_cards[0] if played_cards[0] != voters_card else played_cards[-1]
    )
    thinking = "I can't think"
    prompt_parts = ["I don't have a prompt"]
    return voted_card, thinking, prompt_parts


class RandomPlayer(VisualAllusionsPlayer):
  """A random Visual Allusions player that chooses cards and clues randomly."""

  def __init__(self, player_id: str, clue_space: Optional[List[str]] = None):
    super().__init__(player_id)
    if clue_space is None:
      self.clue_space = [
          "this is a picture",
          "this is a painting",
          "this is a drawing",
          "this is a photograph",
      ]
    else:
      self.clue_space = clue_space

  def update_history(self, scores: Any, votes: Any):
    print(
        f"RandomPlayer {self.player_id} updated history with scores"
        f" {scores}, votes {votes}"
    )

  def choose_card_and_provide_clue(self, hand: list[str]):
    chosen_card = random.choice(hand)
    clue = random.choice(self.clue_space)
    thinking = "I can't think"
    prompt_parts = ["I don't have a prompt"]
    return chosen_card, clue, thinking, prompt_parts

  def choose_card_to_play(
      self,
      hand: list[str],
      clue: str,
      storyteller_player_id: str,
  ):
    chosen_card = random.choice(hand)
    thinking = "I can't think"
    prompt_parts = ["I don't have a prompt"]
    return chosen_card, thinking, prompt_parts

  def vote(
      self,
      played_cards: list[str],
      clue: str,
      voters_card: str,
      storyteller_player_id: str,
  ):
    cards_to_choose = [card for card in played_cards if card != voters_card]
    voted_card = random.choice(cards_to_choose)
    thinking = "I can't think"
    prompt_parts = ["I don't have a prompt"]
    return voted_card, thinking, prompt_parts


class LLMPlayer(VisualAllusionsPlayer):
  """A Visual Allusions player that uses a Large Language Model (LLM) to make decisions."""

  def __init__(
      self,
      player_id,
      model_name="gemini/gemini-2.5-flash",
      temperature=0.7,
      max_output_tokens=4096,
      thinking_budget=1024,
      num_players=4,
      structured_outputs=True,
      card_data_dir=constants.CARDS_DATA_DIR,
      game_instructions=constants.VISUAL_ALLUSIONS_RULES,
      maintain_chat_history=False,
      trim_messages=False,
      max_prompt_tokens=40960,
  ):
    """Initializes an LLM-based Player using the Gemini API.

    Args:
        player_id (str): The unique identifier for the player.
        model_name (str): The name of the Gemini model to use.
        temperature (float): The temperature for the model's generation.
        max_output_tokens (int): The maximum number of output tokens for the
          model.
        thinking_budget (int): The maximum number of tokens for the thinking
          process. If -1, dynamic thinking is used.
        num_players (int): The total number of players in the game.
        structured_outputs (bool): Whether to use structured outputs.
        card_data_dir (str): The directory where card images are stored.
        game_instructions (str): The game instructions.
        maintain_chat_history (bool): Whether to maintain a chat history.
        trim_messages (bool): Whether to trim messages. Recommended use when
          maintaining chat history.
        max_prompt_tokens (int): The maximum number of tokens for the prompt.
          Used for trimming messages.
    """
    super().__init__(player_id)
    self.model_name = model_name
    self.game_instructions = game_instructions
    self.temperature = temperature
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.structured_outputs = (
        structured_outputs if "gemma" not in model_name else False
    )
    self.maintain_chat_history = maintain_chat_history
    self.trim_messages = trim_messages
    self.max_prompt_tokens = max_prompt_tokens
    self.card_data_dir = card_data_dir
    self.num_players = num_players
    self.round_number = 0
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
    # Exclude the unpickleable gemini_client
    del state["llm_client"]
    return state

  def __setstate__(self, state: Any):
    """Restore state from the unpickled state values."""
    self.__dict__.update(state)
    # Re-initialize the gemini_client
    self.initialize_llm_client()

  def _get_system_prompt(self):
    """Defines the system prompt for the LLM player."""

    return (
        f"You are Player {self.player_id} in a game of Visual Allusions."
        f" \n{self.game_instructions}\n."
    )

  def display_player_scores(self):
    """Gets the scores of all players in the game and formats to a string."""
    score_display = ""
    if self.players_scores:

      for player_id, score in self.players_scores.items():
        if player_id == self.player_id:
          score_display += f"Player {player_id} (You) - Score: {score}\n"
        else:
          score_display += f"Player {player_id} - Score: {score}\n"

    else:
      score_display += "This is round 1 of the game. Everyone has 0 scores.\n"

    return score_display

  def _prepare_storyteller_prompt(self, hand: list[str]):
    """Prepares the prompt for the LLM to act as a storyteller.

    Args:
        hand: The storyteller's hand of cards (list of file paths).

    Returns:
        tuple: A tuple containing the prompt parts (list of str) and a card map
          (dict).
    """

    prompt_parts = [
        core_utils.wrap_text_parts(
            f"This is the round {self.round_number} of the game. You are the"
            " storyteller now."
            "\nThese are the scores for all the players currently:"
            f" {self.display_player_scores()}"
            "Your hand of cards is shown below. Each card is labeled with a"
            " letter.\n\n"
        )
    ]

    card_map = core_utils.get_card_map(hand)

    for label, card_filename in card_map.items():
      card_filepath = os.path.join(self.card_data_dir, card_filename)
      prompt_parts.append(core_utils.wrap_text_parts(f"\nCard {label}:\n"))
      prompt_parts.append(core_utils.wrap_image_parts(card_filepath))
      prompt_parts.append(
          core_utils.wrap_text_parts(f"End of the image for Card {label}\n\n")
      )

    prompt_parts.append(
        core_utils.wrap_text_parts(
            "Choose one card from your hand and provide a creative clue for it"
            " (a sentence, a story, or a poem). The clue should be related to"
            " the card but not so obvious that everyone guesses it, and not so"
            " obscure that no one guesses it. Your goal is for some, but not"
            " all, other players to guess your card correctly.\n\nThink"
            " step-by-step about the best card and clue, enclosing your"
            " thoughts in <think> and </think> tags before providing your final"
            " answer in the specified format.\n\nRespond in the following"
            " format:\nChosen Card: [Card Label (Integer), e.g., 0]\nClue:"
            " [Your creative clue]"
        )
    )

    return prompt_parts, card_map

  def _parse_storyteller_response(
      self, response_text: str, response_thinking: Optional[str] = None
  ):
    """Parses the LLM's response when acting as a storyteller.

    Args:
        response_text: The response text from the LLM.
        response_thinking: The thinking process from the LLM if it exists.

    Returns:
        tuple: A tuple containing the chosen card label, the clue, and the
        thinking process.
    """
    if self.structured_outputs:
      # Extract thinking process and final answer
      response_json = json.loads(response_text)

      # Parse the response
      chosen_card_label = response_json.get("storyteller_card")
      clue = response_json.get("clue")
      thinking = response_json.get("thinking")

    else:
      response_text = response_text.replace("[", "").replace("]", "")
      response_text = response_text.replace("**", "")
      # Parse the response
      # Extract the thinking
      thinking_match = re.search(
          r"<think>(.*?)</think>", response_text, re.DOTALL
      )
      thinking = thinking_match.group(1).strip() if thinking_match else ""

      # Extract the chosen card

      card_match = re.search(r"Chosen Card: (\d+)", response_text)
      chosen_card_label = int(card_match.group(1)) if card_match else None

      if chosen_card_label is None:
        # Try again and look for Chosen Card: Card {idx}
        card_match = re.search(r"Chosen Card:\s*Card\s*(\d+)", response_text)
        chosen_card_label = int(card_match.group(1)) if card_match else None

      # Extract the clue
      clue_match = re.search(r"Clue: (.*)", response_text)
      clue = clue_match.group(1).strip() if clue_match else ""

    if response_thinking is not None:
      thinking = (
          f"Internal thinking: {response_thinking}\n\nResponse thinking:"
          f" {thinking}"
      )
    elif thinking:
      thinking = f"Response thinking: {thinking}"

    return chosen_card_label, clue, thinking

  def choose_card_and_provide_clue(
      self,
      hand: list[str],
      prompt_parts: list[str] | None = None,
      card_map: dict[str, str] | None = None,
  ):
    """Storyteller chooses a card and provides a clue via the LLM.

    Args:
        hand: The storyteller's hand of cards (list of file paths).
        prompt_parts: The prompt parts to use for the LLM.
        card_map: The card map to use for the LLM.

    Returns:
        tuple: A tuple containing the chosen card (file path) and the clue
        (str).
               Returns (None, None) if an error occurs or parsing fails.
    """
    if not hand:
      raise ValueError("Storyteller hand is empty.")

    if prompt_parts is None or card_map is None:
      prompt_parts, card_map = self._prepare_storyteller_prompt(hand)

    messages = core_utils.create_messages(
        user_message_parts=prompt_parts,
        system_prompt=self._get_system_prompt(),
        chat_history=self.chat_history,
    )
    response_text, response_thinking = self.llm_client.generate(
        messages=messages,
        response_format=types.StorytellerOutput
        if self.structured_outputs
        else None,
    )

    if self.maintain_chat_history:
      self.chat_history = core_utils.update_history(
          assistant_response=response_text,
          user_message_parts=prompt_parts,
          chat_history=self.chat_history,
      )

    chosen_card_label, clue, thinking = self._parse_storyteller_response(
        response_text, response_thinking
    )

    if chosen_card_label is not None and clue:
      chosen_card_filename = card_map.get(chosen_card_label)
      if chosen_card_filename:
        return (
            chosen_card_filename,
            clue,
            thinking,
            prompt_parts,
        )
      else:
        raise ValueError(f"LLM chose invalid card label '{chosen_card_label}'.")
    else:
      raise ValueError(
          "Could not parse LLM response for chosen card and clue. Raw"
          f" response: {response_text}"
      )

  def _prepare_choose_play_card_prompt(
      self, hand: list[str], clue: str, storyteller_player_id: str
  ):
    """Prepares the prompt for the LLM to choose a card to play.

    Args:
        hand: The player's hand of cards (list of file paths).
        clue: The storyteller's clue for the round.
        storyteller_player_id: The player id of the storyteller.

    Returns:
        tuple: A tuple containing the prompt parts (list of str) and a card map
          (dict).
    """
    prompt_parts = [
        core_utils.wrap_text_parts(
            f"This is the round {self.round_number} of the game. Now is your"
            " turn to play a card given the storyteller's clue.\nThese are the"
            " scores for all the players currently:"
            f" {self.display_player_scores()}The storyteller's (Player"
            f" {storyteller_player_id}) clue is: '{clue}'\n\nYour hand of cards"
            " is shown below. Each card is labeled with a number.\n\n"
        )
    ]

    card_map = core_utils.get_card_map(hand)

    for label, card_filename in card_map.items():
      card_filepath = os.path.join(self.card_data_dir, card_filename)
      prompt_parts.append(core_utils.wrap_text_parts(f"\nCard {label}:\n"))
      prompt_parts.append(core_utils.wrap_image_parts(card_filepath))
      prompt_parts.append(
          core_utils.wrap_text_parts(f"End of the image for Card {label}\n\n")
      )

    prompt_parts.append(
        core_utils.wrap_text_parts(
            "Choose the card from your hand that you think best matches this"
            " clue. Your goal is to make other players think your card is the"
            " storyteller's card.\n\nThink step-by-step about which card in"
            " your hand is the best fit for the clue, enclosing your thoughts"
            " in <think> and </think> tags before providing your final answer"
            " in the specified format.\n\nRespond in the following"
            " format:\nMy chosen card to play is: [Label (Integer), e.g., 0]"
        )
    )

    return prompt_parts, card_map

  def _parse_choose_play_card_response(
      self, response_text: str, response_thinking: Optional[str] = None
  ):
    """Parses the LLM's response when choosing a card to play.

    Args:
        response_text: The response text from the LLM.
        response_thinking: The thinking process from the LLM if it exists.

    Returns:
        tuple: A tuple containing the chosen card label and the thinking
        process.
    """
    if self.structured_outputs:
      response_json = json.loads(response_text)
      chosen_card_label = response_json.get("played_card")
      thinking = response_json.get("thinking")
    else:
      response_text = response_text.replace("[", "").replace("]", "")
      response_text = response_text.replace("**", "")

      thinking_match = re.search(
          r"<think>(.*?)</think>", response_text, re.DOTALL
      )
      thinking = thinking_match.group(1).strip() if thinking_match else ""

      chosen_card_match = re.search(
          r"My chosen card to play is:\s*([0-9])", response_text
      )
      chosen_card_label = (
          int(chosen_card_match.group(1)) if chosen_card_match else None
      )

      if chosen_card_label is None:
        # Try again and look for Chosen Card: Card {idx}
        card_match = re.search(
            r"My chosen card to play is:\s*Card\s*(\d+)", response_text
        )
        chosen_card_label = int(card_match.group(1)) if card_match else None

    if response_thinking is not None:
      thinking = (
          f"Internal thinking: {response_thinking}\n\nResponse thinking:"
          f" {thinking}"
      )
    elif thinking:
      thinking = f"Response thinking: {thinking}"

    return chosen_card_label, thinking

  def choose_card_to_play(
      self,
      hand: list[str],
      clue: str,
      storyteller_player_id: str,
      prompt_parts: list[str] | None = None,
      card_map: dict[str, str] | None = None,
  ):
    """Guesser chooses a card from their hand to play using the LLM.

    Args:
        hand: The player's hand of cards (list of file paths).
        clue: The storyteller's clue for the round.
        storyteller_player_id: The player id of the storyteller.
        prompt_parts: The prompt parts to use for the LLM.
        card_map: The card map to use for the LLM.

    Returns:
        str: The file path of the chosen card from the hand.
    """

    if not hand:
      raise ValueError("Guesser hand is empty.")

    if prompt_parts is None or card_map is None:
      prompt_parts, card_map = self._prepare_choose_play_card_prompt(
          hand, clue, storyteller_player_id
      )

    messages = core_utils.create_messages(
        user_message_parts=prompt_parts,
        system_prompt=self._get_system_prompt(),
        chat_history=self.chat_history,
    )
    response_text, response_thinking = self.llm_client.generate(
        messages=messages,
        response_format=types.CardPlayOutput
        if self.structured_outputs
        else None,
    )

    if self.maintain_chat_history:
      self.chat_history = core_utils.update_history(
          assistant_response=response_text,
          user_message_parts=prompt_parts,
          chat_history=self.chat_history,
      )

    chosen_card_label, thinking = self._parse_choose_play_card_response(
        response_text, response_thinking
    )

    if chosen_card_label is not None:
      chosen_card_filename = card_map.get(chosen_card_label)
      if chosen_card_filename:
        return (
            chosen_card_filename,
            thinking,
            prompt_parts,
        )
      else:
        raise ValueError(
            f"LLM chose invalid card label '{chosen_card_label}' for playing a"
            " card."
        )
    else:
      raise ValueError(
          "Could not parse LLM response for chosen card to play. Raw response:"
          f" {response_text}"
      )

  def _prepare_vote_prompt(
      self,
      played_cards: list[str],
      clue: str,
      voters_card: str,
      storyteller_player_id: str,
  ):
    """Prepares the prompt for the LLM to vote for the storyteller's card.

    Args:
        played_cards: A list of the file paths of the cards played in the round
          (including the storyteller's, shuffled).
        clue: The storyteller's clue for the round.
        voters_card: The file path of the card the player is voting for.
        storyteller_player_id: The player id of the storyteller.

    Returns:
        tuple: A tuple containing the prompt parts (list of str) and a card map
          (dict).
    """
    prompt_parts = [
        core_utils.wrap_text_parts(
            f"This is the round {self.round_number} of the game. You are now in"
            " the voting phase of this round.\nThese are the scores for all"
            f" the players currently: {self.display_player_scores()}The"
            f" storyteller's (Player {storyteller_player_id}) clue is:"
            f" '{clue}'\n\nThe cards played this round are shown below. Each"
            " card is labeled with a number. You need to vote for the card you"
            " believe is the storyteller's original card.\n\n"
        )
    ]

    # Create a map from label to card path for played cards
    played_card_map = core_utils.get_card_map(played_cards)
    for label, card_filename in played_card_map.items():
      card_filepath = os.path.join(self.card_data_dir, card_filename)
      prompt_parts.append(core_utils.wrap_text_parts(f"\nOption {label}:\n"))
      prompt_parts.append(core_utils.wrap_image_parts(card_filepath))
      prompt_parts.append(
          core_utils.wrap_text_parts(f"End of the image for Option {label}\n\n")
      )

    self_played_card_label = next(
        label for label, path in played_card_map.items() if path == voters_card
    )
    prompt_parts.append(
        core_utils.wrap_text_parts(
            f"\nRemember, Option {self_played_card_label} is YOUR card. Do NOT"
            " vote for your own card."
        )
    )
    prompt_parts.append(
        core_utils.wrap_text_parts(
            "Vote for the card you believe is the storyteller's original card,"
            " based on the clue.Think step-by-step about which played card best"
            " matches the clue and which one the storyteller likely chose,"
            " enclosing your thoughts in <think> and </think> tags before"
            " providing your final answer in the specified format.\n\nRespond"
            " in the following format:\nI vote for card: [Label (Integer),"
            " e.g., Card 0]"
        )
    )

    return prompt_parts, played_card_map

  def _parse_vote_response(
      self, response_text: str, response_thinking: Optional[str] = None
  ):
    """Parses the LLM's response when voting for the storyteller's card.

    Args:
        response_text: The response text from the LLM.
        response_thinking: The thinking process from the LLM if it exists.

    Returns:
        tuple: A tuple containing the chosen card label and the thinking
        process.
    """
    if self.structured_outputs:
      response_json = json.loads(response_text)
      voted_card_label = response_json.get("voted_card")
      thinking = response_json.get("thinking")
    else:
      response_text = response_text.replace("[", "").replace("]", "")
      response_text = response_text.replace("**", "")
      thinking_match = re.search(
          r"<think>(.*?)</think>", response_text, re.DOTALL
      )
      thinking = thinking_match.group(1).strip() if thinking_match else ""
      voted_card_match = re.search(r"I vote for card:\s*([0-9])", response_text)
      voted_card_label = (
          int(voted_card_match.group(1)) if voted_card_match else None
      )

      if voted_card_label is None:
        # Try again and look for Chosen Card: Card {idx}
        card_match = re.search(
            r"I vote for card:\s*Card\s*(\d+)", response_text
        )
        voted_card_label = int(card_match.group(1)) if card_match else None

    if response_thinking is not None:
      thinking = (
          f"Internal thinking: {response_thinking}\n\nResponse thinking:"
          f" {thinking}"
      )
    elif thinking:
      thinking = f"Response thinking: {thinking}"

    return voted_card_label, thinking

  def vote(
      self,
      played_cards: list[str],
      clue: str,
      voters_card: str,
      storyteller_player_id: str,
      prompt_parts: list[str] | None = None,
      played_card_map: dict[str, str] | None = None,
  ):
    """Guesser votes for the storyteller's card using the LLM.

    Args:
        played_cards: A list of the file paths of the cards played in the round
          (including the storyteller's, shuffled).
        clue: The storyteller's clue for the round.
        voters_card: The file path of the card the player is voting for.
        storyteller_player_id: The player id of the storyteller.
        prompt_parts: The prompt parts to use for the LLM.
        played_card_map (dict): The card map to use for the LLM.

    Returns:
        str: The file path of the card the player is voting for.
             Returns None if an error occurs or parsing fails.
    """

    if not played_cards:
      raise ValueError("Played cards is empty.")

    if prompt_parts is None or played_card_map is None:
      prompt_parts, played_card_map = self._prepare_vote_prompt(
          played_cards, clue, voters_card, storyteller_player_id
      )

    messages = core_utils.create_messages(
        user_message_parts=prompt_parts,
        system_prompt=self._get_system_prompt(),
        chat_history=self.chat_history,
    )
    response_text, response_thinking = self.llm_client.generate(
        messages=messages,
        response_format=types.VoteOutput if self.structured_outputs else None,
    )

    if self.maintain_chat_history:
      self.chat_history = core_utils.update_history(
          assistant_response=response_text,
          user_message_parts=prompt_parts,
          chat_history=self.chat_history,
      )

    voted_card_label, thinking = self._parse_vote_response(
        response_text, response_thinking
    )

    if voted_card_label is not None:
      voted_card_filename = played_card_map.get(voted_card_label)
      if voted_card_filename:
        return (
            voted_card_filename,
            thinking,
            prompt_parts,
        )
      else:
        raise ValueError(
            f"LLM voted for invalid card label '{voted_card_label}'."
        )
    else:
      raise ValueError(
          "Could not parse LLM response for voted card. Raw response:"
          f" {response_text}"
      )

  def __call__(
      self, observation: spaces.AgentObservation
  ) -> tuple[spaces.AgentAction, list[Any] | None]:

    self.round_number = observation.round_number
    self.current_role = observation.current_role

    return super().__call__(observation)


class LLMPlayerWithSharedContext(LLMPlayer):
  """An LLM-based Visual Allusions player with shared context and full history."""

  def __init__(
      self,
      player_id,
      partner_player_id,
      shared_context_stories,
      model_name="gemini/gemini-2.5-flash",
      temperature=0.7,
      max_output_tokens=4096,
      thinking_budget=1024,
      num_players=4,
      structured_outputs=True,
      card_data_dir=constants.CARDS_DATA_DIR,
      game_instructions=constants.VISUAL_ALLUSIONS_RULES,
      maintain_chat_history=False,
      trim_messages=False,
      max_prompt_tokens=40960,
  ):
    """Initializes an LLM-based Player with shared context and full history."""
    super().__init__(
        player_id=player_id,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_budget=thinking_budget,
        num_players=num_players,
        structured_outputs=structured_outputs,
        card_data_dir=card_data_dir,
        game_instructions=game_instructions,
        maintain_chat_history=maintain_chat_history,
        trim_messages=trim_messages,
        max_prompt_tokens=max_prompt_tokens,
    )
    self.partner_player_id = partner_player_id
    self.shared_context_stories = shared_context_stories

  def _get_shared_context_for_prompt(self):
    """Formats the shared context stories for inclusion in the prompt."""
    if not self.shared_context_stories:
      return []

    context_prompt_parts = [
        (
            f"You (Player {self.player_id}) and Player"
            f" {self.partner_player_id} are fans of stories from a short"
            " stories magazine called Tell Me A Story. Your favorite stories"
            " are provided below. You can reference elements from these"
            " stories in your clues and card choices when acting as"
            f" storyteller, such that when Player {self.partner_player_id} is"
            " able to understand the reference but not others."
        ),
        (
            "\nYou can also refer to these stories while trying to understand"
            f" the clue given by Player {self.partner_player_id} when it is"
            " their turn to be the storyteller.\n\n###### SHARED CONTEXT"
            " STORIES #######"
        ),
        "\n\n",
    ]

    for _, story in enumerate(self.shared_context_stories):
      context_prompt_parts.append(
          f"\n{story}\n\n"  # Include the full story text
      )
      context_prompt_parts.append(
          "----------------------------------------\n\n"
      )

    context_prompt_parts.append("#########################################")
    context_prompt_parts.append("\n\n")

    return context_prompt_parts

  def _get_system_prompt(self):
    """Defines the system prompt for the LLM player with shared context."""
    base_prompt = (
        super()._get_system_prompt()
    )  # Get the base system prompt from the parent class
    shared_context_prompt = self._get_shared_context_for_prompt()

    # Combine base prompt and shared context prompt
    system_prompt_parts = [base_prompt] + shared_context_prompt
    return "".join(system_prompt_parts)


def initialize_player(
    player_id: str,
    player_type: str,
    card_data_dir: str = constants.CARDS_DATA_DIR,
    game_instructions: str = constants.VISUAL_ALLUSIONS_RULES,
    llm_args: Optional[types.LLMPlayerArgs] = None,
    partner_player_id: Optional[str] = None,
    shared_context_stories: Optional[List[str]] = None,
) -> VisualAllusionsPlayer:
  """Initializes a player of the specified type.

  Args:
      player_id: The ID of the player.
      player_type: The type of player to initialize ("simple", "random", or
        "llm").
      card_data_dir: Directory containing card images.
      game_instructions: Instructions for the game.
      llm_args: LLM generation arguments, required for "llm" player type.
      partner_player_id: Partner player ID for shared context LLM player.
      shared_context_stories: Shared context stories for shared context LLM
        player.

  Returns:
      An instance of a VisualAllusionsPlayer.
  """
  if player_type == "simple":
    return SimplePlayer(player_id)
  elif player_type == "random":
    return RandomPlayer(player_id)
  elif player_type == "llm":
    if llm_args is None:
      llm_args = types.LLMPlayerArgs()
    if partner_player_id and shared_context_stories:
      return LLMPlayerWithSharedContext(
          player_id=player_id,
          partner_player_id=partner_player_id,
          shared_context_stories=shared_context_stories,
          card_data_dir=card_data_dir,
          game_instructions=game_instructions,
          model_name=llm_args.model_name,
          temperature=llm_args.temperature,
          max_output_tokens=llm_args.max_output_tokens,
          thinking_budget=llm_args.thinking_budget,
          num_players=llm_args.num_players,
          structured_outputs=llm_args.structured_outputs,
          maintain_chat_history=llm_args.maintain_chat_history,
          trim_messages=llm_args.trim_messages,
          max_prompt_tokens=llm_args.max_prompt_tokens,
      )
    else:
      return LLMPlayer(
          player_id=player_id,
          card_data_dir=card_data_dir,
          game_instructions=game_instructions,
          model_name=llm_args.model_name,
          temperature=llm_args.temperature,
          max_output_tokens=llm_args.max_output_tokens,
          thinking_budget=llm_args.thinking_budget,
          num_players=llm_args.num_players,
          structured_outputs=llm_args.structured_outputs,
          maintain_chat_history=llm_args.maintain_chat_history,
          trim_messages=llm_args.trim_messages,
          max_prompt_tokens=llm_args.max_prompt_tokens,
      )
  else:
    raise ValueError(f"Unknown player type: {player_type}")
