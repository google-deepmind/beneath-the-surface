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

"""LLM client for the Environments."""

import dataclasses
import json
from typing import Any, List, Optional
import litellm
import pydantic


@dataclasses.dataclass
class LLMClient:
  """Base class for LLM clients."""

  model: str
  temperature: float = 0.7
  max_tokens: int = 4096
  thinking_enabled: bool = True
  thinking_budget_tokens: int = 1024
  trim_messages: bool = False
  max_prompt_tokens: int = 40960

  @staticmethod
  def get_thinking_from_response(
      response: litellm.types.utils.ModelResponse,
  ) -> Optional[str]:
    """Extracts the thinking trace from the model response."""
    if not response.choices:
      return None

    # Attempt 1: Check if reasoning_content exists in the response.
    try:
      thinking_trace = response.choices[0].message.reasoning_content
    except AttributeError:
      thinking_trace = None

    if thinking_trace:
      return thinking_trace

    # Attempt 2: Check if thinking_blocks exists in the response.
    try:
      thinking_blocks = response.choices[0].message.thinking_blocks
      thinking_trace = "\n".join(
          [json.loads(block["thinking"])["text"] for block in thinking_blocks]
      )
      return thinking_trace
    except AttributeError:
      return None

  def generate(
      self,
      messages: List[Any],
      response_format: Optional[pydantic.BaseModel] = None,
      **kwargs,
  ) -> tuple[str, Optional[str]]:
    """Generate a response from the LLM.

    Args:
        messages: List of message dictionaries
        response_format: Optional pydantic.BaseModel to format the response
          with.
        **kwargs: Additional parameters to pass to the LLM

    Returns:
        Generated text response
    """
    if self.trim_messages:
      messages = litellm.utils.trim_messages(
          messages, max_tokens=self.max_prompt_tokens
      )

    response = litellm.completion(
        model=self.model,
        messages=messages,
        response_format=response_format,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        thinking={
            "type": "enabled" if self.thinking_enabled else "disabled",
            "budget_tokens": self.thinking_budget_tokens,
        },
        drop_params=True,
        **kwargs,
    )
    response_text = response.choices[0].message.content
    thinking_trace = self.get_thinking_from_response(response)
    return response_text, thinking_trace
