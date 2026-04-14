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

"""Utility functions for the core subtext library."""

import base64
import copy
import json
import os
import random
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image


def load_file(file_path):
  """Loads a file from the specified directory."""
  with open(file_path, "r") as f:
    return f.read()


def load_image(image_path):
  return Image.open(open(image_path, "rb"))


def load_image_bytes(image_path):
  """Loads an image and returns it as a byte string."""
  with open(image_path, "rb") as f:
    return f.read()


def get_image_url(image_path):
  """Returns the URL of an image to use with LiteLLM API for vision inputs."""
  image_bytes = load_image_bytes(image_path)
  b64_image = base64.b64encode(image_bytes).decode("utf-8")
  return f"data:image/jpeg;base64,{b64_image}"


def wrap_image_parts(image_path):
  """Loads an image and returns it in a format suitable for the Gemini API."""
  image_url = get_image_url(image_path)
  return {
      "type": "image_url",
      "image_url": {"url": image_url},
  }


def wrap_text_parts(text):
  """Wraps text in a format suitable for the Gemini API."""
  return {
      "type": "text",
      "text": text,
  }


def create_messages(
    user_message_parts: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
):
  """Creates a list of messages for the Gemini API."""
  messages = []
  if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
  if chat_history:
    messages.extend(chat_history)
  messages.append({"role": "user", "content": user_message_parts})
  return messages


def update_history(
    assistant_response: str,
    user_message_parts: List[Dict[str, Any]],
    chat_history: List[Dict[str, Any]],
):
  """Updates the chat history with the assistant and user messages."""

  updated_chat_history = copy.deepcopy(chat_history)
  updated_chat_history.append({"role": "user", "content": user_message_parts})
  updated_chat_history.append(
      {"role": "assistant", "content": assistant_response}
  )
  return updated_chat_history


def set_seed(seed):
  """Sets the random seed for reproducibility."""
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)


def load_tms_data(tms_dir, split="validation"):
  """Loads Tell Me a Story (TMS) data from the specified split.

  Args:
      tms_dir: The directory containing the Tell Me a Story data.
      split: The split of the data to load (e.g., "train", "validation").

  Returns:
      A pandas DataFrame containing the TMS data.
  """
  rows = []
  with open(f"{tms_dir}/tell-me-a-story-{split}.jsonl", "r") as f:
    for line in f:
      example = json.loads(line)
      rows.append(example)
  tms_df = pd.DataFrame(rows)
  tms_df = tms_df.rename(columns={"targets": "story"})
  tms_df["num_words"] = tms_df["story"].apply(lambda x: len(x.split(" ")))
  return tms_df


def load_shared_context_data(
    sc_data_dir,
    max_story_len=5000,
    num_stories=10,
    seed=42,
    **kwargs,
):
  """Loads shared context data from the specified source.

  Args:
      sc_data_dir: The directory containing the shared context data.
      max_story_len: The maximum length of a story in words.
      num_stories: The number of stories to load.
      seed: The seed for random sampling.
      **kwargs: Additional keyword arguments.

  Returns:
      A pandas DataFrame containing the shared context data.
  """
  sc_df = load_tms_data(sc_data_dir, split=kwargs.get("split", "validation"))

  # Select stories that are within the maximum length limit
  sc_df = sc_df[sc_df["num_words"] <= max_story_len]

  # Select random stories from the remaining stories
  if len(sc_df) >= num_stories:
    sc_df = sc_df.sample(n=num_stories, random_state=seed, replace=False)
  else:
    print(
        f"Warning: Only {len(sc_df)} stories found with less than"
        f" {max_story_len} words. Using all of them."
    )
    sc_df = sc_df["story"].tolist()

  return sc_df["story"].tolist()


def load_cards_data(data_dir, get_prompts=False):
  """Loads card image file names from the specified directory.

  Args:
      data_dir: The directory containing the card images.
      get_prompts: Whether to return the prompts as well.

  Returns:
      A list of card image file names.
  """

  card_files = [
      f
      for f in os.listdir(data_dir)
      if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")
  ]
  print(f"Found {len(card_files)} image files in {data_dir}")
  if get_prompts:
    with open(f"{data_dir}/prompts.jsonl", "r") as f:
      prompts = f.read().split("\n")
      return card_files, prompts

  else:
    return card_files


def load_prompts_and_vanilla_clues(data_dir):
  """Loads prompts and vanilla clues from the specified directory.

  Args:
      data_dir: The directory containing the prompts file.

  Returns:
      A tuple containing a list of prompts and a list of vanilla clues.
  """
  with open(f"{data_dir}/prompts.jsonl", "r") as f:
    prompts = f.read().split("\n")

  vanilla_clues = [
      prompt.replace("Please create", "").strip() for prompt in prompts
  ]

  return prompts, vanilla_clues


def get_partnerships(exp_config):
  """Extracts partnerships from the experiment config."""
  partners = {}
  for player in exp_config["players"]:
    if player.get("partner_player_id"):
      partners[player["player_id"]] = player["partner_player_id"]
  return partners


def get_card_map(card_paths):
  return {label: path for label, path in enumerate(card_paths)}
