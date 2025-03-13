import time

from flare_ai_consensus.consensus.consensus import run_consensus
from flare_ai_consensus.router import AsyncOpenRouterProvider
from flare_ai_consensus.settings import Message, settings
from flare_ai_consensus.utils import load_json

from flare_ai_consensus.embeddings import EmbeddingModel
from flare_ai_consensus.price_utils import FTSOFeed, WalletBalances

async def run_consensus_test(
    message: str, wallet_address: str
) -> tuple[str, dict, dict, list[float]]:

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

    # We use the last 8 hours to make everything a bit faster
    from_ts = int(time.time()) - 3600 * 8
    to_ts = int(time.time())

    feed = FTSOFeed()
    wallet = WalletBalances()

    balances = wallet.get_balances(wallet_address)
    flr_usd_feed = feed.get_feed_analytics("FLR/USD", from_ts, to_ts)
    usdc_usd_feed = feed.get_feed_analytics("USDC/USD", from_ts, to_ts)

    print(balances)

    initial_conversations: list[list[Message]] = []
    for i in range(len(settings.consensus_config.models)):
        initial_conversations.append(
            [
                {
                    "role": "system",
                    "content": settings.consensus_config.models[i].system_prompt,
                },
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": f"These are the user's wallet balances: {balances}. These are the current price feeds with statistics: {flr_usd_feed}, {usdc_usd_feed}",
                }
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
