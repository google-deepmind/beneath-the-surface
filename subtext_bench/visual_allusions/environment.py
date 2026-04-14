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

"""Environment for the Visual Allusions game."""

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from subtext_bench.visual_allusions import constants
from subtext_bench.visual_allusions import rendering
from subtext_bench.visual_allusions import results
from subtext_bench.visual_allusions import spaces
from subtext_bench.visual_allusions import types

VisualAllusionsDeck = types.VisualAllusionsDeck
VisualAllusionsGameState = spaces.VisualAllusionsGameState
AgentRole = types.AgentRole
AgentObservation = spaces.AgentObservation
AgentAction = spaces.AgentAction
AgentInfo = results.AgentInfo


class VisualAllusionsEnv:
  """A self-contained Visual Allusions environment class that mimics the PettingZoo API."""

  metadata = {"render_modes": ["human", "html"], "name": "visual_allusions_v0"}

  def __init__(
      self,
      card_image_paths: List[str],
      card_data_dir: str = constants.DATA_DIR,
      num_players: int = 4,
      winning_score: int = 30,
      agent_names: Optional[List[str]] = None,
      log_path: str = "visual_allusions_game_log.log",
  ):
    # --- PettingZoo API properties ---
    if agent_names is None:
      self.possible_agents = [f"player_{i}" for i in range(num_players)]
    else:
      if len(agent_names) != num_players:
        raise ValueError(
            f"Must provide a name for each player. {len(agent_names)} provided"
            f" but {num_players} players are required."
        )
      self.possible_agents = agent_names
    self.agent_selection = ""
    self.num_players_observed = 0

    # --- Game-specific properties ---
    self._card_image_paths = card_image_paths
    self._num_players = num_players
    self._winning_score = winning_score
    # self.game_state: Optional[VisualAllusionsGameState] = None

    # --- Rendering properties ---
    self.renderer = rendering.VisualAllusionsRenderer(
        self, card_data_dir, log_path
    )
    # self.AgentRole = AgentRole

    # Directories
    self.card_data_dir = card_data_dir

    # Reset the game state
    self.reset()

  def observe(self, agent: str) -> AgentObservation:
    """Returns an observation for the given agent.

    Args:
        agent: The name of the agent for which to return the observation.

    Returns:
        An AgentObservation object containing the agent's current
        observation.
    """
    agent_idx = self.agent_turn_mapping[agent]
    gs = self.game_state
    role = self._get_agent_role(agent_idx)

    voting_deck = None
    played_card = None
    if role in [AgentRole.VOTER, AgentRole.OBSERVER]:
      # Provide only the card paths for voting, not who owns them
      voting_deck = [card_path for card_path, _ in gs.cards_in_play]
      for card_path, own_card_idx in gs.cards_in_play:
        if own_card_idx == agent_idx:
          played_card = card_path
          break

      if played_card is None:
        raise ValueError(
            f"Played card not found for voter {agent} in cards_in_play"
        )

    votes = None
    storyteller_card = None

    if role == AgentRole.OBSERVER:
      votes = gs.player_votes
      storyteller_card = gs.storyteller_card

    return AgentObservation(
        round_number=gs.round_number,
        current_role=role,
        hand=gs.player_hands[agent_idx],
        clue=gs.storyteller_clue,
        voting_deck=voting_deck,
        played_card=played_card,
        storyteller_player_id=self.agent_turns[gs.storyteller_index],
        game_scores=gs.player_scores,
        last_round_scores=gs.last_round_scores,
        personal_score=gs.player_scores[self.agent_turns[agent_idx]],
        votes=votes,
        storyteller_card=storyteller_card,
    )

  def reset(
      self,
      game_state: Optional[VisualAllusionsGameState] = None,
      seed: Optional[int] = None,
      ignore_position_bias: bool = False,
  ):
    """Resets the environment to its initial state.

    Args:
        game_state: Optional game state to restore the environment to.
        seed: Optional seed for the random number generator.
        ignore_position_bias: Whether to ignore position bias in the agent turn
          order.
    """
    np.random.seed(seed)
    self.agents = self.possible_agents[:]
    self.rewards = {agent: 0 for agent in self.agents}
    self.terminations = {agent: False for agent in self.agents}
    self.truncations = {agent: False for agent in self.agents}
    self.infos: Dict[str, AgentInfo] = {
        agent: AgentInfo() for agent in self.agents
    }
    self.sparks: Dict[Tuple[str, str], int] = {}
    self.round_summary: Optional[Dict[str, Any]] = None
    self.round_summary_to_display: Optional[Dict[str, Any]] = None
    self.done = False
    self.previous_agent = None
    self.num_players_observed = 0
    # We randomize the turn order to prevent position bias
    if ignore_position_bias:
      self.agent_turns = [str(agent_name) for agent_name in self.agents]
    else:
      self.agent_turns = [
          str(agent_name)
          for agent_name in list(np.random.permutation(self.agents))
      ]
    self.agent_turn_mapping = {
        agent: i for i, agent in enumerate(self.agent_turns)
    }

    deck = VisualAllusionsDeck(self._card_image_paths)
    if game_state is not None:
      self.game_state = game_state
    else:
      self.game_state = VisualAllusionsGameState(
          players=self.agents,
          deck=deck,
          winning_score=self._winning_score,
      )

    # Set the first agent to act
    self._set_next_agent()

  def last(self):
    """Returns the observation, reward, and other info for the current agent."""
    agent = self.agent_selection
    observation = self.observe(agent)
    reward = self.rewards[agent]
    termination = self.terminations[agent]
    truncation = self.truncations[agent]
    done = self.done
    info = self.infos[agent]
    self.infos[agent].last_observation = observation
    self.infos[agent].all_observations.append(observation)
    return observation, reward, termination, truncation, done, info

  def step(self, action: AgentAction):
    """Executes a step in the environment based on the provided action.

    This method takes an action from the current agent, updates the game state
    accordingly, calculates rewards, and prepares for the next agent's turn.

    Args:
        action: An AgentAction object containing the agent's chosen action.
    """
    if (
        self.terminations[self.agent_selection]
        or self.truncations[self.agent_selection]
    ):
      self._was_dead_step()
      return

    current_agent_name = self.agent_selection
    agent_idx = self.agent_turn_mapping[current_agent_name]

    # Reset rewards and info at the start of each agent's turn
    for agent in self.agents:
      self.rewards[agent] = 0

    # --- Process action based on game stage ---
    stage = self.game_state.round_stage

    if stage == "storytelling":
      self._storyteller_step(agent_idx, action)
      self._prepare_guesser_card_play_stage()
    elif stage == "guesser_card_play":
      self._guesser_card_play_step(agent_idx, action)
      if len(self.game_state.cards_in_play) == self._num_players:
        self._prepare_guesser_vote_stage()
    elif stage == "guesser_vote":
      self._guesser_vote_step(agent_idx, action)
      if len(self.game_state.player_votes) == self._num_players - 1:
        self._score_round()
        self._prepare_observe_only_stage()
    elif stage == "observe_only":
      self._observe_only_step()
      if self.num_players_observed == self._num_players:
        self._prepare_next_round()

    self.infos[current_agent_name].last_action = action
    self.infos[current_agent_name].all_actions.append(action)

    if not self.done:
      # Set the next agent to act
      self._set_next_agent()

  def render(self):
    """Renders the current state of the environment."""
    context = self._get_render_context()
    self.renderer.render(context)

  def _get_render_context(self):
    """Gathers the necessary information for rendering."""
    gs = self.game_state
    if not gs:
      return None
    current_role = self._get_agent_role(
        self.agent_turn_mapping[self.agent_selection]
    )

    # if current_role == AgentRole.OBSERVER:
    #   return None

    context = {
        "game_state": gs,
        "previous_agent_info": None,
        "previous_agent_role": None,
        "round_summary": None,
        "done": self.done,
        "current_agent_name": self.agent_selection,
        "role": current_role,
        "cards_to_display": [],
        "display_title": "",
    }

    if (
        self.previous_agent
        and self.previous_agent in self.infos
        and self.infos[self.previous_agent].last_action
        and self.infos[self.previous_agent].last_observation
    ):
      context["previous_agent_info"] = self.infos[self.previous_agent]
      last_observation = self.infos[self.previous_agent].last_observation
      context["previous_agent_role"] = (
          last_observation.current_role
          if last_observation is not None
          else None
      )

    if (
        self.round_summary_to_display
    ):  # and self._last_round_scored < gs.round_number - 1:
      context["round_summary"] = self.round_summary_to_display
      self.round_summary_to_display = None  # Clear after getting

    if context["role"] == AgentRole.VOTER:
      context["display_title"] = "Cards on table to vote for:"
      context["cards_to_display"] = [
          card_path for card_path, _ in gs.cards_in_play
      ]
    elif context["role"] in [
        AgentRole.STORYTELLER,
        AgentRole.CARDPLAYER,
    ]:
      agent_idx = self.agent_turn_mapping[self.agent_selection]
      context["display_title"] = f"{self.agent_selection}'s hand:"
      context["cards_to_display"] = gs.player_hands[agent_idx]

    return context

  def checkpoint(self):
    """Returns a deepcopy of the current environment state."""
    return copy.deepcopy(self)

  def restore(self, checkpoint):
    """Restores the environment state from a checkpoint."""
    self.__dict__.update(checkpoint.__dict__)

  def run_single_round(self, player_policies):
    """Runs a single round of the game from the current state."""
    initial_round_number = self.game_state.round_number
    # The environment is already set to the desired state (via restore).
    # We just need to run the loop until the round is over.
    while (
        self.game_state.round_number == initial_round_number and not self.done
    ):
      obs, _, _, _, _, _ = self.last()

      # Get the current agent
      current_agent = self.agent_selection
      policy = player_policies[current_agent]

      action = policy(obs)
      self.step(action)

    return self.game_state.player_scores

  def close(self):
    """Closes any open resources, like the HTML file handler."""
    self.renderer.close()

  # --- Helper Methods ---

  def _was_dead_step(self):
    """Handles a step() call for an agent that is already terminated.

    Its job is to simply advance to the next agent.
    """
    self._set_next_agent()

  def _set_next_agent(self):
    """Sets the next agent to act based on the current game state.

    This method determines the next agent in the turn order, considering
    the current round stage (storytelling, card playing, or voting) and
    the current storyteller. It updates `self.agent_selection` to the
    name of the next agent.
    """
    gs = self.game_state
    current_turn_order = []

    self.previous_agent = self.agent_selection

    if gs.round_stage == "storytelling":
      current_turn_order = [gs.storyteller_index]
    elif gs.round_stage in ["guesser_card_play", "guesser_vote"]:
      # All players except the storyteller act in order
      current_turn_order = [
          (gs.storyteller_index + i) % gs.num_players
          for i in range(1, gs.num_players)
      ]

    if gs.round_stage == "storytelling":
      next_agent_idx = gs.storyteller_index
    elif gs.round_stage == "guesser_card_play":
      num_cards_played_by_guessers = len(gs.cards_in_play) - 1
      next_agent_idx = current_turn_order[num_cards_played_by_guessers]
    elif gs.round_stage == "guesser_vote":
      num_votes_cast = len(gs.player_votes)
      next_agent_idx = current_turn_order[num_votes_cast]
    elif gs.round_stage == "observe_only":
      next_agent_idx = self.num_players_observed
    else:
      raise ValueError(f"Unknown round stage: {gs.round_stage}")

    self.agent_selection = self.agent_turns[next_agent_idx]

  def _get_agent_role(self, agent_idx: int) -> Optional[AgentRole]:
    """Determines the current role of an agent based on the game state.

    Args:
        agent_idx: The index of the agent in the current turn order.

    Returns:
        The AgentRole of the specified agent, or None if the agent is
        not currently active.
    """
    gs = self.game_state
    is_current_agent = self.agent_selection == self.agent_turns[agent_idx]

    if not is_current_agent:
      return None  # Agent is waiting

    if gs.round_stage == "storytelling":
      return AgentRole.STORYTELLER
    elif gs.round_stage == "guesser_card_play":
      return AgentRole.CARDPLAYER
    elif gs.round_stage == "guesser_vote":
      return AgentRole.VOTER
    elif gs.round_stage == "observe_only":
      return AgentRole.OBSERVER
    return None

  def _storyteller_step(self, agent_idx: int, action: AgentAction):
    gs = self.game_state
    card = action.card_to_play
    gs.storyteller_clue = action.clue
    gs.cards_in_play.append((card, agent_idx))
    gs.player_hands[agent_idx].remove(card)
    gs.storyteller_card = card

  def _prepare_guesser_card_play_stage(self):
    self.game_state.round_stage = "guesser_card_play"

  def _guesser_card_play_step(self, agent_idx: int, action: AgentAction):
    gs = self.game_state
    card = action.card_to_play
    if card is None:
      raise ValueError("card_to_play cannot be None")
    gs.cards_in_play.append((card, agent_idx))
    gs.player_hands[agent_idx].remove(card)

  def _prepare_guesser_vote_stage(self):
    self.game_state.round_stage = "guesser_vote"
    random.shuffle(self.game_state.cards_in_play)

  def _guesser_vote_step(self, agent_idx: int, action: AgentAction):
    card = action.card_to_vote
    if card is None:
      raise ValueError("card_to_vote cannot be None")
    self.game_state.player_votes[self.agent_turns[agent_idx]] = card

  def _prepare_observe_only_stage(self):
    self.game_state.round_stage = "observe_only"
    self.num_players_observed = 0

  def _observe_only_step(self):
    self.num_players_observed += 1

  def _score_round(self):
    """Scores the current round, updates scores, and prepares the next round.

    This method calculates scores based on how many players voted for the
    storyteller's card, updates the game state with these scores.
    """
    gs = self.game_state
    storyteller_idx = gs.storyteller_index
    storyteller_card_path = ""
    for card, owner in gs.cards_in_play:
      if owner == storyteller_idx:
        storyteller_card_path = card
        break

    votes_for_storyteller_card = 0
    votes_for_guesser_cards = {p_idx: 0 for p_idx in range(gs.num_players)}
    successful_comm_pairs = []
    for voter, voted_card in gs.player_votes.items():
      if voted_card == storyteller_card_path:
        votes_for_storyteller_card += 1
        successful_comm_pairs.append(
            (storyteller_idx, self.agent_turn_mapping[voter])
        )
      else:
        for card_path, owner_idx in gs.cards_in_play:
          if card_path == voted_card:
            votes_for_guesser_cards[owner_idx] += 1
            break

    round_scores = {player: 0 for player in self.agents}
    num_guessers = gs.num_players - 1
    if (
        votes_for_storyteller_card == 0
        or votes_for_storyteller_card == num_guessers
    ):
      # Storyteller gets 0 points.
      gs.player_granular_scores[self.agent_turns[storyteller_idx]][
          "storytelling"
      ] += 0
      for i in range(gs.num_players):
        if i != storyteller_idx:
          round_scores[self.agent_turns[i]] = 2
          gs.player_granular_scores[self.agent_turns[i]]["bonus"] += 2
    else:
      round_scores[self.agent_turns[storyteller_idx]] = 3
      gs.player_granular_scores[self.agent_turns[storyteller_idx]][
          "storytelling"
      ] += 3
      for voter, voted_card in gs.player_votes.items():
        if voted_card == storyteller_card_path:
          round_scores[voter] += 3
          gs.player_granular_scores[voter]["guessing"] += 3

    for guesser_idx, num_votes in votes_for_guesser_cards.items():
      if guesser_idx != storyteller_idx:
        round_scores[self.agent_turns[guesser_idx]] += num_votes
        gs.player_granular_scores[self.agent_turns[guesser_idx]][
            "distractor"
        ] += num_votes

    for player in self.agents:
      gs.player_scores[player] += round_scores[player]
      self.rewards[player] = round_scores[player]
    gs.last_round_scores = copy.deepcopy(round_scores)

    storyteller_agent = self.agent_turns[storyteller_idx]
    self.infos[storyteller_agent].previous_votes = gs.player_votes
    self.infos[storyteller_agent].previous_cards_in_play = [
        card for card, _ in gs.cards_in_play
    ]

    self.round_summary = {
        "storyteller_card": storyteller_card_path,
        "storyteller_clue": gs.storyteller_clue,
        "votes": dict(gs.player_votes),
        "cards_in_play": list(gs.cards_in_play),
        "round_scores": round_scores,
        "total_scores": gs.player_scores,
    }
    self.round_summary_to_display = copy.deepcopy(self.round_summary)

    # Store some statistics that might be useful for later analysis
    storyteller = self.agent_turns[storyteller_idx]
    ## First we will classify the type of clue given by the storyteller
    if votes_for_storyteller_card == 0:
      clue_type = "obscure"
    elif votes_for_storyteller_card == num_guessers:
      clue_type = "obvious"
    else:
      clue_type = "just-right"
    self.infos[storyteller].clue_types.append(clue_type)

    ## Now we will update the spark between different players
    for pair in successful_comm_pairs:
      pair_key = (self.agent_turns[pair[0]], self.agent_turns[pair[1]])
      if pair_key not in self.sparks:
        self.sparks[pair_key] = 0
      self.sparks[pair_key] += 1

    ## We will store the number of storytelling rounds for each agent
    self.infos[storyteller].num_storytelling_rounds += 1

  def _prepare_next_round(self):
    """Sets up the environment for the next round."""
    gs = self.game_state

    gs.discard_pile.extend([card for card, _ in gs.cards_in_play])
    gs.reset_round_state()
    for i in range(gs.num_players):
      if gs.deck:
        gs.player_hands[i].extend(gs.deck.deal_cards(1))

    gs.round_number += 1
    gs.rotate_storyteller()
    gs.round_stage = "storytelling"

    if gs.is_game_over():
      for agent in self.agents:
        self.terminations[agent] = True
      self.done = True
