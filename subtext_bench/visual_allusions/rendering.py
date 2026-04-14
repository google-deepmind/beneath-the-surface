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

"""Rendering logic for the Visual Allusions game environment."""

import base64
import io
import logging
import os

from subtext_bench.core import utils as core_utils
from subtext_bench.visual_allusions import types


class VisualAllusionsRenderer:
  """Handles rendering the Visual Allusions game state."""

  def __init__(self, env, card_data_dir, log_path=None):
    self._env = env
    self.card_data_dir = card_data_dir
    self._log_path = log_path
    self._html_log_path = None
    self._html_file_handler = None
    self._last_round_scored = 0

    self.logger = logging.getLogger(f"VisualAllusionsRenderer-{id(self)}")
    self.logger.setLevel(logging.INFO)
    self.logger.propagate = False  # Don't pass messages to parent loggers

    if self.logger.hasHandlers():
      self.logger.handlers.clear()

    # Always log to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    self.logger.addHandler(stream_handler)

    if self._log_path:
      # Create text log file handler
      log_file_path = os.path.splitext(self._log_path)[0] + ".log"
      file_handler = logging.FileHandler(log_file_path, mode="w")
      file_handler.setFormatter(logging.Formatter("%(message)s"))
      self.logger.addHandler(file_handler)

      # Create HTML log file path if requested via render_modes
      if "html" in self._env.metadata["render_modes"]:
        self._html_log_path = os.path.splitext(self._log_path)[0] + ".html"

  def _get_card_as_html(self, card_path: str, caption: str = "") -> str:
    """Encodes a card image into a base64 string for HTML embedding."""
    full_path = os.path.join(self.card_data_dir, card_path)
    # Use gfile to open the image file
    img = core_utils.load_image(full_path)

    img.thumbnail((300, 300))  # Resize for reasonable file size

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG", quality=85)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    return (
        '<div style="display: inline-block; margin: 10px; text-align:'
        ' center; border: 1px solid #ccc; padding: 5px; border-radius: 5px;">'
        f'<img src="data:image/jpeg;base64,{img_base64}"'
        f' alt="{caption}" style="width:150px;">'
        f"<p>{caption}</p></div>"
    )

  def _render_human(self, context):
    """Renders the current game state to the console or an IPython display."""
    if not context:
      return

    gs = context["game_state"]

    if context["previous_agent_info"]:
      prev_info = context["previous_agent_info"]
      prev_obs = prev_info.last_observation
      prev_action = prev_info.last_action

      if prev_info.last_prompt:
        self.logger.info("\n--- %s LLM Prompt ---", self._env.previous_agent)
        for part in prev_info.last_prompt:
          if isinstance(part, str):
            self.logger.info(part)
          else:
            self.logger.info("[Image Data]")
        self.logger.info("--- End Prompt ---")

      if prev_action.thinking_trace:
        self.logger.info(
            "%s Thinking Trace: %s",
            self._env.previous_agent,
            prev_action.thinking_trace,
        )

      if prev_obs and prev_obs.current_role == types.AgentRole.STORYTELLER:
        self.logger.info(
            "%s as storyteller chose the card: %s",
            self._env.previous_agent,
            prev_action.card_to_play,
        )
        self.logger.info(
            "%s as storyteller gave the clue: %s",
            self._env.previous_agent,
            prev_action.clue,
        )
      elif prev_obs and prev_obs.current_role == types.AgentRole.CARDPLAYER:
        self.logger.info(
            "%s as cardplayer played the card: %s",
            self._env.previous_agent,
            prev_action.card_to_play,
        )
      elif prev_obs and prev_obs.current_role == types.AgentRole.VOTER:
        self.logger.info(
            "%s as voter voted for the card: %s",
            self._env.previous_agent,
            prev_action.card_to_vote,
        )

    if context["round_summary"]:
      round_summary = context["round_summary"]
      self._last_round_scored = gs.round_number - 1

      self.logger.info("\n--- Round Results ---")
      # Simplified console output for round results
      self.logger.info(
          "Storyteller's clue: %s", round_summary["storyteller_clue"]
      )
      self.logger.info("Scores this round: %s", round_summary["round_scores"])
      self.logger.info("---------------------")

    if context["done"]:
      self.logger.info("\n%s", "=" * 40)
      self.logger.info("Game Over | SCORES: %s", gs.player_scores)
      return

    if context["role"] == types.AgentRole.OBSERVER:
      return

    self.logger.info("\n%s", "=" * 40)
    self.logger.info(
        "ROUND %s | STAGE: %s | SCORES: %s",
        gs.round_number,
        gs.round_stage.replace("_", " ").upper(),
        gs.player_scores,
    )
    self.logger.info(
        "Storyteller is %s", self._env.agent_turns[gs.storyteller_index]
    )
    self.logger.info(
        "--> Current Turn: %s (Role: %s)",
        context["current_agent_name"].upper(),
        context["role"].value if context["role"] else "Waiting",
    )

    if gs.storyteller_clue:
      self.logger.info("The Clue is: '%s'", gs.storyteller_clue)

    if context["cards_to_display"]:
      self.logger.info("\n%s", context["display_title"])
      for i, card_path in enumerate(context["cards_to_display"]):
        caption = (
            "Option %s" % i
            if context["role"] == types.AgentRole.VOTER
            else "Card %s" % i
        )
        self.logger.info("- %s: %s", caption, os.path.basename(card_path))

    print("=" * 40 + "\n")

  def _render_html(self, context):
    """Appends a detailed, snapshot of the current game state to an HTML log."""
    if self._html_file_handler is None:
      self._html_file_handler = open(
          self._html_log_path, "w", encoding="utf-8"
      )
      self._html_file_handler.write(
          "<html><head><title>Visual Allusions Game"
          " Log</title><style>body {font-family:"
          " sans-serif; margin: 2em;} h1, h2, h3"
          " {border-bottom: 2px solid"
          " #ddd; padding-bottom: 5px;} h2"
          " {margin-top: 2em;} .turn-separator"
          " {border-top: 3px double #000;"
          " margin-top: 2em;}.card-container"
          " {display: flex; flex-wrap: wrap;"
          " align-items: flex-start;}</style>"
          "</head><body><h1>Visual Allusions"
          " Game Log</h1>"
      )
      self._last_round_scored = -1

    f = self._html_file_handler
    if not context:
      return

    gs = context["game_state"]

    if (
        context["previous_agent_info"]
        and context["previous_agent_role"]
        and context["previous_agent_role"] != types.AgentRole.OBSERVER
    ):
      prev_info = context["previous_agent_info"]
      f.write(
          "<h4>Action</h4><p style='font-style: italic; background-color:"
          " #f4f4f4; padding: 5px; border-radius: 4px;'>"
      )
      if prev_info.last_action.thinking_trace:
        f.write(
            f"<b>{self._env.previous_agent}'s Thought:</b>"
            f" {prev_info.last_action.thinking_trace}<br>"
        )

      prev_obs = prev_info.last_observation
      prev_action = prev_info.last_action
      if prev_obs and prev_obs.current_role == types.AgentRole.STORYTELLER:
        f.write(
            f"<b>{self._env.previous_agent}</b> as storyteller chose the card:"
            f" {prev_action.card_to_play}<br>"
        )
        f.write(
            f"<b>{self._env.previous_agent}</b> as storyteller gave the clue:"
            f" <b>'{prev_action.clue}'</b>"
        )
      elif prev_obs and prev_obs.current_role == types.AgentRole.CARDPLAYER:
        f.write(
            f"<b>{self._env.previous_agent}</b> as cardplayer played the card:"
            f" {prev_action.card_to_play}"
        )
      elif prev_obs and prev_obs.current_role == types.AgentRole.VOTER:
        f.write(
            f"<b>{self._env.previous_agent}</b> as voter voted for the card:"
            f" {prev_action.card_to_vote}"
        )
      f.write("</p>")

    if context["round_summary"]:
      round_summary = context["round_summary"]
      self._last_round_scored = gs.round_number - 1
      votes_on_card = {card: [] for card, _ in round_summary["cards_in_play"]}
      for voter, voted_card in round_summary["votes"].items():
        if voted_card in votes_on_card:
          votes_on_card[voted_card].append(voter)

      f.write("<h2>Round Results</h2><div class='card-container'>")
      for card_path, owner_idx in round_summary["cards_in_play"]:
        caption = f"Played by {self._env.agent_turns[owner_idx]}<br>"
        if card_path == round_summary["storyteller_card"]:
          caption += "<b>Storyteller's Card ✅</b>"
        voters = votes_on_card.get(card_path, [])
        if voters:
          caption += (
              f"<br><small>Votes from: {', '.join(sorted(voters))}</small>"
          )
        f.write(self._get_card_as_html(card_path, caption=caption))
      f.write("</div>")
      f.write(
          f"<p><b>Scores this round:</b> {round_summary['round_scores']}</p>"
      )
      f.write(f"<p><b>Total Scores:</b> {round_summary['total_scores']}</p>")

    if context["done"]:
      f.write(
          f"<h2>Game Over</h2><p><b>Final Scores:</b> {gs.player_scores}</p>"
      )
      f.flush()
      return

    if context["role"] == types.AgentRole.OBSERVER:
      f.flush()
      return

    f.write('<div class="turn-separator"></div>')
    f.write(
        f"<h2>ROUND {gs.round_number} | STAGE:"
        f" {gs.round_stage.replace('_', ' ').upper()}</h2>"
    )
    f.write(
        f"<p><b>Scores:</b> {gs.player_scores}<br><b>Storyteller:</b>"
        f" {self._env.agent_turns[gs.storyteller_index]}</p>"
    )
    f.write(
        f"<h3>--> Current Turn: {context['current_agent_name'].upper()} (Role:"
        f" {context['role'].value if context['role'] else 'Waiting'})</h3>"
    )

    if gs.storyteller_clue:
      f.write(f"<h4>The Clue is: '{gs.storyteller_clue}'</h4>")

    if context["cards_to_display"]:
      f.write(
          f"<h4>{context['display_title']}</h4><div class='card-container'>"
      )
      for i, card_path in enumerate(context["cards_to_display"]):
        caption = (
            f"Option {i}"
            if context["role"] == types.AgentRole.VOTER
            else f"Card {i}"
        )
        f.write(
            self._get_card_as_html(
                card_path,
                caption=(
                    f"{caption}<br><small>{os.path.basename(card_path)}</small>"
                ),
            )
        )
      f.write("</div>")

    f.flush()

  def render(self, context):
    """Renders the current state of the environment."""
    if "human" in self._env.metadata["render_modes"]:
      self._render_human(context)
    if "html" in self._env.metadata["render_modes"]:
      self._render_html(context)

  def close(self):
    """Closes any open resources, like the HTML file handler."""
    if self._html_file_handler:
      if self._env.done:
        self._html_file_handler.write('<div class="turn-separator"></div>')
        self._html_file_handler.write("<h2>Game Over</h2>")
        self._html_file_handler.write(
            f"<p><b>Final Scores:</b> {self._env.game_state.player_scores}</p>"
        )

      self._html_file_handler.write("</body></html>")
      self._html_file_handler.close()
      self._html_file_handler = None
      self.logger.info("HTML log saved to %s", self._html_log_path)
