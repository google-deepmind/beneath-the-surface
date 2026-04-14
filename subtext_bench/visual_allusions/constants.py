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

"""Constants for the Visual Allusions game."""

DATA_DIR = "data/visual_allusions/"
CARDS_DATA_DIR = f"{DATA_DIR}/cards/"
TMS_DATA_DIR = "data/tell_me_a_story"

VISUAL_ALLUSIONS_RULES = (
    "Visual Allusions is a creative guessing game played with illustrated"
    " cards. The game proceeds in rounds, with a different player being the"
    " 'storyteller' each round. Each player has a hand of"
    " cards.\n\n**Storyteller's Turn:**\nThe storyteller chooses a card from"
    " their hand and comes up with a clue (a sentence, a story or a poem) that"
    " relates to that card. The clue should be creative and not too obvious,"
    " but also not so obscure that no one can guess it. The storyteller places"
    " their chosen card face down.\n\n**Other Players' Turn:**\nEach other"
    " player looks at the clue and chooses a card from their own hand that they"
    " feel best matches the clue. They also place their chosen card face"
    " down.\n\n**Reveal and Voting:**\nAll played cards are shuffled and then"
    " revealed face up. Each non-storyteller player then secretly votes for the"
    " card they believe was the storyteller's original card.\n\n**Scoring:**\n-"
    " If all non-storyteller players guess the storyteller's card correctly, OR"
    " if none of the non-storyteller players guess correctly, the storyteller"
    " scores 0 points, and all other players score 2 points.- If some, but not"
    " all, non-storyteller players guess the storyteller's card correctly, the"
    " storyteller scores 3 points, and each player who guessed correctly scores"
    " 3 points.- Players other than the storyteller score 1 point for each vote"
    " their *own* played card receives (if it wasn't the storyteller's"
    " card).\n\n**End of Round:**\nPlayers draw new cards to replenish their"
    " hand size (usually 6 or 7 cards). The role of the storyteller rotates to"
    " the next player.\n\n**Game End:**\nThe game typically ends when the deck"
    " runs out of cards or when a player reaches a predetermined score"
    " limit.\n\nYour goal is to play strategically according to these rules to"
    " maximize your score. "
)
