from flare_ai_consensus.consensus.consensus import run_consensus
from flare_ai_consensus.router import AsyncOpenRouterProvider
from flare_ai_consensus.settings import Message, settings
from flare_ai_consensus.utils import load_json

from flare_ai_consensus.embeddings import EmbeddingModel


async def run_consensus_test(
    message: str,
) -> tuple[str, dict, dict, float]:

    # Load input configuration.
    config_json = load_json(settings.input_path / "input.json")
    settings.load_consensus_config(config_json)

    # Initialize the OpenRouter provider.
    provider = AsyncOpenRouterProvider(
        api_key=settings.open_router_api_key, base_url=settings.open_router_base_url
    )

    embedding_model = EmbeddingModel(
        api_key=settings.gemini_embedding_key, base_url=settings.open_router_base_url
    )

    initial_conversations: list[list[Message]] = []
    for i in range(len(settings.consensus_config.models)):
        initial_conversations.append(
            [
                {
                    "role": "system",
                    "content": settings.consensus_config.models[i].system_prompt,
                },
                {"role": "user", "content": message},
            ]
        )

    # Run consensus algorithm
    answer, shapley_values, response_data, confidence = await run_consensus(
        provider,
        settings.consensus_config,
        initial_conversations,
        embedding_model,
    )

    return answer, shapley_values, response_data, confidence
