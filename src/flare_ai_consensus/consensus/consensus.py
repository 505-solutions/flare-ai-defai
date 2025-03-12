import asyncio
import math
import time

import structlog

from flare_ai_consensus.consensus.aggregator import (
    async_centralized_embedding_aggregator
)
from flare_ai_consensus.embeddings import EmbeddingModel
from flare_ai_consensus.router import AsyncOpenRouterProvider, ChatRequest
from flare_ai_consensus.settings import ConsensusConfig, Message, ModelConfig
from flare_ai_consensus.utils import parse_chat_response
from flare_ai_consensus.ftso_feed import FTSOFeed

logger = structlog.get_logger(__name__)


async def run_consensus(
        provider: AsyncOpenRouterProvider,
        consensus_config: ConsensusConfig,
        initial_conversation: list[list[Message]],
        embedding_model: EmbeddingModel
) -> tuple[str, dict, dict, float]:

    response_data = {}
    response_data["initial_conversation"] = initial_conversation

    weighted_shapley_values = {}
    total_weight = 0

    # to make everything faster
    from_ts = int(time.time()) - 3600 * 8
    to_ts = int(time.time())

    feed = FTSOFeed()
    flr_usd_feed = feed.get_feed_analytics("FLR/USD", from_ts, to_ts)
    usdc_usd_feed = feed.get_feed_analytics("USDC/USD", from_ts, to_ts)

    for conversation in initial_conversation:
        conversation.append(
            {
                "role": "assistant",
                "content": f"Price feeds: {flr_usd_feed}, {usdc_usd_feed}",
            }
        )

    # Step 1: Initial round.
    responses = await send_round(
        provider, consensus_config, response_data["initial_conversation"]
    )
    aggregated_response, shapley_values, _ = await async_centralized_embedding_aggregator(
        embedding_model, responses
    )

    initial_weight = 1.0  # Base weight for first iteration
    weighted_shapley_values = {k: v * initial_weight for k, v in shapley_values.items()}
    total_weight = initial_weight

    logger.info(
        "initial response aggregation complete", aggregated_response=aggregated_response
    )

    response_data["iteration_0"] = responses
    response_data["aggregate_0"] = aggregated_response
    response_data["shapley_0"] = shapley_values

    confidence = 0

    # Step 2: Improvement rounds.
    for i in range(consensus_config.iterations):
        decay_factor = 1
        iteration_weight = math.exp(-decay_factor * (i + 1))

        responses = await send_round(
            provider, consensus_config, initial_conversation, aggregated_response
        )

        aggregated_response, shapley_values, confidence = await async_centralized_embedding_aggregator(
            embedding_model, responses
        )

        for k, v in shapley_values.items():
            if k in weighted_shapley_values:
                weighted_shapley_values[k] += v * iteration_weight
            else:
                weighted_shapley_values[k] = v * iteration_weight

        total_weight += iteration_weight

        logger.info(
            "responses aggregated",
            iteration=i + 1,
            aggregated_response=aggregated_response,
            iteration_weight=iteration_weight
        )

        response_data[f"iteration_{i + 1}"] = responses
        response_data[f"aggregate_{i + 1}"] = aggregated_response
        response_data[f"shapley_{i + 1}"] = shapley_values
        response_data[f"weight_{i + 1}"] = iteration_weight

    normalized_shapley_values = {k: v / total_weight for k, v in weighted_shapley_values.items()}

    return aggregated_response, normalized_shapley_values, response_data, confidence


def _build_improvement_conversation(
    model: ModelConfig,
    consensus_config: ConsensusConfig,
    initial_conversation: list[Message],
    aggregated_response: str,
) -> list[Message]:
    """Build an updated conversation using the consensus configuration.

    :param consensus_config: An instance of ConsensusConfig.
    :param initial_conversation: the input user prompt with system instructions.
    :param aggregated_response: The aggregated consensus response.
    :return: A list of messages for the updated conversation.
    """
    conversation = initial_conversation.copy()

    # Add aggregated response
    conversation.append(
        {
            "role": consensus_config.aggregated_prompt_type,
            "content": f"Consensus: {aggregated_response}",
        }
    )

    # Add new prompt as "user" message
    conversation.append(
        {"role": "user", "content": consensus_config.improvement_prompt if model.improvement_prompt is None else model.improvement_prompt}
    )

    return conversation


async def _get_response_for_model(
    provider: AsyncOpenRouterProvider,
    consensus_config: ConsensusConfig,
    model: ModelConfig,
    initial_conversation: list[Message],
    aggregated_response: str | None,
) -> tuple[str | None, str]:
    """
    Asynchronously sends a chat completion request for a given model.

    :param provider: An instance of an asynchronous OpenRouter provider.
    :param consensus_config: An instance of ConsensusConfig.
    :param model: A ModelConfig instance.
    :param initial_conversation: the input user prompt with system instructions.
    :param aggregated_response: The aggregated consensus response
        from the previous round (or None).
    :return: A tuple of (model_id, response text).
    """
    if not aggregated_response:
        # Use initial prompt for the first round.
        conversation = initial_conversation
        logger.info("sending initial prompt", model_id=model.model_id)
    else:
        # Build the improvement conversation.
        conversation = _build_improvement_conversation(
            model, consensus_config, initial_conversation, aggregated_response
        )
        logger.info("sending improvement prompt", model_id=model.model_id)

    payload: ChatRequest = {
        "model": model.model_id,
        "messages": conversation,
        "max_tokens": model.max_tokens,
        "temperature": model.temperature,
    }
    response = await provider.send_chat_completion(payload)
    text = parse_chat_response(response)
    logger.info("new response", model_id=model.model_id, response=text)
    return model.model_id, text


async def send_round(
    provider: AsyncOpenRouterProvider,
    consensus_config: ConsensusConfig,
    initial_conversation: list[list[Message]],
    aggregated_response: str | None = None,
) -> dict:
    """
    Asynchronously sends a round of chat completion requests for all models.

    :param provider: An instance of an asynchronous OpenRouter provider.
    :param consensus_config: An instance of ConsensusConfig.
    :param initial_conversation: the input user prompt with system instructions.
    :param aggregated_response: The aggregated consensus response from the
        previous round (or None).
    :return: A dictionary mapping model IDs to their response texts.
    """
    tasks = [
        _get_response_for_model(
            provider, consensus_config, model, initial_conversation[i] if len(consensus_config.models) == len(initial_conversation) else initial_conversation[0], aggregated_response
        )
        for i, model in enumerate(consensus_config.models)
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)
