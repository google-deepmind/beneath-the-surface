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

"""Spark coefficient calculation for Visual Allusions."""

import numpy as np
from scipy import stats


def get_spark_uni_a_b(sparks_dict, player_a, player_b, players):
  """Computes unidirectional spark from player_a to player_b."""
  num_clues_by_a = sparks_dict["storyteller_counts"].get(player_a, 0)
  if num_clues_by_a == 0:
    return 0.0

  num_clues_b_guessed = sparks_dict["unsym_sparks"].get((player_a, player_b), 0)
  other_players = [p for p in players if p not in [player_a, player_b]]
  num_clues_players_guessed = [
      sparks_dict["unsym_sparks"].get((player_a, player), 0)
      for player in other_players
  ]

  player_b_guess_rate = num_clues_b_guessed / num_clues_by_a
  other_players_guess_rate = [
      num_clues_guessed / num_clues_by_a
      for num_clues_guessed in num_clues_players_guessed
  ]

  mean_other_rate = (
      np.mean(other_players_guess_rate) if other_players_guess_rate else 0.0
  )

  return player_b_guess_rate - mean_other_rate


def get_spark_a_b(sparks_dict, player_a, player_b, players):
  """Computes symmetric spark between player_a and player_b."""
  spark_uni_ab = get_spark_uni_a_b(sparks_dict, player_a, player_b, players)
  spark_uni_ba = get_spark_uni_a_b(sparks_dict, player_b, player_a, players)

  spark_coefficient = (spark_uni_ab + spark_uni_ba) / 2
  return spark_coefficient


def chi_squared_test_spark(
    sparks_dict, all_players, player_a_name, player_b_name
):
  """Perform chi-squared test for player pair."""
  sparks_data = sparks_dict["unsym_sparks"]
  storyteller_counts = sparks_dict["storyteller_counts"]

  # Test 1: Is Player B significantly better at guessing Player A's clues?
  total_clues_a = storyteller_counts.get(player_a_name, 0)
  is_a_test_significant = False
  is_b_best_guesser = False
  p_value_a = 1.0
  if total_clues_a > 0:
    guessers_for_a = [p for p in all_players if p != player_a_name]
    contingency_table_a = []
    guesser_names_a = []
    scores_a = []
    for guesser in guessers_for_a:
      key = (player_a_name, guesser)
      correct_guesses = sparks_data.get(key, 0)
      incorrect_guesses = total_clues_a - correct_guesses
      contingency_table_a.append([correct_guesses, incorrect_guesses])
      guesser_names_a.append(guesser)
      scores_a.append(correct_guesses)
    try:
      if (
          contingency_table_a
          and np.sum(contingency_table_a) > 0
          and len(contingency_table_a) > 1
      ):
        _, p_value_a, _, _ = stats.chi2_contingency(contingency_table_a)
        is_a_test_significant = p_value_a < 0.05
      if player_b_name in guesser_names_a:
        b_score_index = guesser_names_a.index(player_b_name)
        is_b_best_guesser = (
            scores_a[b_score_index] == max(scores_a)
            and scores_a[b_score_index] > 0
        )
    except ValueError:
      pass

  # Test 2: Is Player A significantly better at guessing Player B's clues?
  total_clues_b = storyteller_counts.get(player_b_name, 0)
  is_b_test_significant = False
  is_a_best_guesser = False
  p_value_b = 1.0
  if total_clues_b > 0:
    guessers_for_b = [p for p in all_players if p != player_b_name]
    contingency_table_b = []
    guesser_names_b = []
    scores_b = []
    for guesser in guessers_for_b:
      key = (player_b_name, guesser)
      correct_guesses = sparks_data.get(key, 0)
      incorrect_guesses = total_clues_b - correct_guesses
      contingency_table_b.append([correct_guesses, incorrect_guesses])
      guesser_names_b.append(guesser)
      scores_b.append(correct_guesses)
    try:
      if (
          contingency_table_b
          and np.sum(contingency_table_b) > 0
          and len(contingency_table_b) > 1
      ):
        _, p_value_b, _, _ = stats.chi2_contingency(contingency_table_b)
        is_b_test_significant = p_value_b < 0.05
      if player_a_name in guesser_names_b:
        a_score_index = guesser_names_b.index(player_a_name)
        is_a_best_guesser = (
            scores_b[a_score_index] == max(scores_b)
            and scores_b[a_score_index] > 0
        )
    except ValueError:
      pass

  if is_a_test_significant and is_b_test_significant:
    stat_sig = "\\faCheckDouble"
  elif not (is_a_test_significant or is_b_test_significant):
    stat_sig = "\\faTimes"
  else:
    stat_sig = "\\faCheck"

  return (
      str(is_a_test_significant),
      str(is_b_test_significant),
      stat_sig,
      p_value_a,
      p_value_b,
      str(is_a_best_guesser),
      str(is_b_best_guesser),
  )
