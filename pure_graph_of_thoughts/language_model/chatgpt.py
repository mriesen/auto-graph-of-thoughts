import json
import logging
from typing import Self, cast

import backoff
from openai import OpenAI, RateLimitError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from ..api.language_model import LanguageModelException
from ..api.language_model.language_model import LanguageModel
from ..api.language_model.prompt import Prompt
from ..api.state import State


class ChatGPT(LanguageModel):
    """
    The ChatGPT language model.
    """

    _model_id: str = 'gpt-3.5-turbo-0125'
    _prompt_token_cost: float = 0.5 / 1_000_000
    _completion_token_cost: float = 1.5 / 1_000_000
    _temperature: float = 1.0
    _max_tokens: int = 1536
    _seed: int = 0
    _n_total_prompt_tokens: int = 0
    _n_total_completion_tokens: int = 0
    _client: OpenAI
    _total_cost: float = 0
    _currency: str = '$'
    _logger: logging.Logger

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def currency(self) -> str:
        return self._currency

    def __init__(self, api_key: str) -> None:
        """
        Initializes a new ChatGPT instance.
        :param api_key: OpenAI API key
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._client = OpenAI(
                api_key=api_key
        )

    @backoff.on_exception(
            backoff.expo, RateLimitError, logger=Self.__class__.__name__, max_time=30, max_tries=3, factor=10
    )
    def prompt(self, prompt: Prompt, state: State) -> State:
        """
        Queries the GPT API and returns the output state.
        :param prompt: prompt to use
        :param state: input state to use
        :return: output state
        """
        self._logger.debug('Calling OpenAI API with prompt %s and state %s', prompt, state)
        response: ChatCompletion = self._client.chat.completions.create(
                model=self._model_id,
                messages=[{'role': 'user', 'content': prompt.for_input(state)}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={'type': 'json_object'},
                seed=self._seed,
                n=1
        )
        if (
                response.usage is None
                or response.usage.prompt_tokens is None
                or response.usage.completion_tokens is None
        ):
            raise LanguageModelException('Response usage or prompt/completion tokens is None')

        usage: CompletionUsage = response.usage
        n_prompt_tokens = usage.prompt_tokens
        n_completion_tokens = usage.completion_tokens
        self._n_total_prompt_tokens += n_prompt_tokens
        self._n_total_completion_tokens += n_completion_tokens
        delta_cost = (
                self._prompt_token_cost * n_prompt_tokens
                + self._completion_token_cost * n_completion_tokens
        )
        self._add_cost(delta_cost)
        self._logger.debug(f'Response ChatGPT: %s', response)
        self._logger.debug(
                f'Cost delta / total: %s %s / %s %s',
                delta_cost, self.currency,
                self.total_cost, self.currency
        )
        content = response.choices[0].message.content
        if content is not None:
            return cast(State, json.loads(content.strip()))
        raise LanguageModelException('Response content is None')

    def _add_cost(self, delta_cost: float) -> None:
        self._total_cost += delta_cost
