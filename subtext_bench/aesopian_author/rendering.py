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

"""Rendering logic for The Aesopian Author game environment."""

import logging
import os

from subtext_bench.aesopian_author import spaces


class AesopianAuthorRenderer:
  """Handles rendering The Aesopian Author game state."""

  def __init__(self, env, log_path=None):
    self._env = env
    self._log_path = log_path
    self._html_log_path = None
    self._html_file_handler = None
    self._last_round_scored = 0

    self.logger = logging.getLogger(f"AesopianAuthorRenderer-{id(self)}")
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

  def _render_human(self, context):
    """Renders the current game state to the console."""
    if not context:
      return

    gs = context["game_state"]

    if context["previous_agent_info"]:
      prev_info = context["previous_agent_info"]
      prev_action = prev_info.last_action

      if prev_action.thinking_trace:
        self.logger.info(
            "%s Thinking Trace: %s",
            context["previous_agent"],
            prev_action.thinking_trace,
        )

      if (
          isinstance(prev_action, spaces.AuthorialAction)
          and prev_action.current_story
      ):
        self.logger.info(
            "%s as author wrote the story: %s",
            context["previous_agent"],
            prev_action.current_story,
        )
      if (
          isinstance(prev_action, spaces.InterpretorAction)
          and prev_action.interpretation
      ):
        self.logger.info(
            "%s as interpretor gave the interpretation: %s",
            context["previous_agent"],
            prev_action.interpretation,
        )

    if context["round_summary"]:
      round_summary = context["round_summary"]
      self._last_round_scored = gs.round_number - 1
      self.logger.info("\n--- Round Results ---")
      self.logger.info("Scores this round: %s", round_summary["round_scores"])
      self.logger.info("--------------------")

    if context["done"]:
      self.logger.info("\n%s", "=" * 40)
      self.logger.info("Game Over | SCORES: %s", gs.player_scores)
      return

    self.logger.info("\n%s", "=" * 40)
    self.logger.info(
        "ROUND %s | STAGE: %s | SCORES: %s",
        gs.round_number,
        gs.round_stage.replace("_", " ").upper(),
        gs.player_scores,
    )
    self.logger.info(
        "--> Current Turn: %s (Role: %s)",
        context["current_agent_name"].upper(),
        context["role"].value if context["role"] else "Waiting",
    )
    self.logger.info("%s\n", "=" * 40)

  def _render_html(self, context):
    """Appends a detailed snapshot of the current game state to an HTML log."""
    if self._html_file_handler is None:
      self._html_file_handler = open(self._html_log_path, "w", encoding="utf-8")
      self._html_file_handler.write(
          "<html><head><title>The Aesopian Author Game Log</title><style>"
          "body {font-family: sans-serif; margin: 2em;}"
          "h1, h2, h3 {border-bottom: 2px solid #ddd; padding-bottom: 5px;}"
          "h2 {margin-top: 2em;}"
          ".turn-separator {border-top: 3px double #000; margin-top: 2em;}"
          ".story, .thought, .prompt { white-space: pre-wrap; background-color:"
          " #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid"
          " #dee2e6; margin-top: 1em; }"
          "</style></head><body><h1>The Aesopian Author Game Log</h1>"
      )
      self._last_round_scored = -1

    f = self._html_file_handler
    if not context:
      return

    gs = context["game_state"]

    if context["previous_agent_info"] and context["previous_agent_role"]:
      prev_info = context["previous_agent_info"]

      f.write("<h4>Action</h4>")
      if prev_info.last_action.thinking_trace:
        f.write(f"<b>{context['previous_agent']}'s Thought:</b>")
        f.write(
            f"<div class='thought'>{prev_info.last_action.thinking_trace}</div>"
        )

      prev_action = prev_info.last_action
      if (
          isinstance(prev_action, spaces.AuthorialAction)
          and prev_action.current_story
      ):
        f.write(
            f"<b>{context['previous_agent']}</b> as author wrote the story:"
        )
        f.write(f"<div class='story'>{prev_action.current_story}</div>")
      if (
          isinstance(prev_action, spaces.InterpretorAction)
          and prev_action.interpretation
      ):
        f.write(
            f"<p><b>{context['previous_agent']}</b> as interpretor gave the"
            " interpretation:"
            f" <b>'{prev_action.interpretation}'</b></p>"
        )

    if context["round_summary"]:
      round_summary = context["round_summary"]
      self._last_round_scored = gs.round_number - 1

      f.write("<h2>Round Results</h2>")
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

    if gs.round_stage != "SCORING":
      f.write('<div class="turn-separator"></div>')
      f.write(
          f"<h2>ROUND {gs.round_number} | STAGE:"
          f" {gs.round_stage.replace('_', ' ').upper()}</h2>"
      )
      f.write(f"<p><b>Scores:</b> {gs.player_scores}<br>")
      f.write(
          "<h3>--> Current Turn:"
          f" {context['current_agent_name'].upper()} (Role:"
          f" {context['role'].value if context['role'] else 'Waiting'})</h3>"
      )

      f.flush()

  def render(self, context):
    """Renders the current state of the environment."""
    if "human" in self._env.metadata["render_modes"]:
      self._render_human(context)
    if "html" in self._env.metadata["render_modes"]:
      self._render_html(context)

  def close(self, game_state, metrics):
    """Closes any open resources, like the HTML file handler."""
    if self._html_file_handler:
      self._html_file_handler.write('<div class="turn-separator"></div>')
      self._html_file_handler.write("<h2>Game Over</h2>")
      self._html_file_handler.write(
          f"<p><b>Final Scores:</b> {game_state.player_scores}</p>"
      )
      self._html_file_handler.write("<h3>Metrics</h3>")
      self._html_file_handler.write("<ul>")
      for metric, value in metrics.__dict__.items():
        if not metric.startswith("_"):
          self._html_file_handler.write(
              f"<li><b>{metric.replace('_', ' ').title()}:</b> {value}</li>"
          )
      self._html_file_handler.write("</ul>")

      self._html_file_handler.write("</body></html>")
      self._html_file_handler.close()
      self._html_file_handler = None
      self.logger.info("HTML log saved to %s", self._html_log_path)
