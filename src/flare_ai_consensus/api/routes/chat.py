import json
import time

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from flare_ai_consensus.consensus import run_consensus
from flare_ai_consensus.embeddings import EmbeddingModel
from flare_ai_consensus.router import AsyncOpenRouterProvider
from flare_ai_consensus.settings import ConsensusConfig, Message
from flare_ai_consensus.utils.parser_utils import extract_values
from price_utils import FTSOFeed, WalletBalances

logger = structlog.get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """
    Pydantic model for chat message validation.

    Attributes:
        message (str): The chat message content, must not be empty
    """

    user_message: str = Field(..., min_length=1)

class ChatRouter:
    """
    A simple chat router that processes incoming messages using the CL pipeline.
    """

    def __init__(
            self,
            router: APIRouter,
            provider: AsyncOpenRouterProvider,
            embedding_model: EmbeddingModel,
            consensus_config: ConsensusConfig | None = None,
    ) -> None:
        """
        Initialize the ChatRouter.

        Args:
            router (APIRouter): FastAPI router to attach endpoints.
            provider: instance of an async OpenRouter client.
            consensus_config: config for running the consensus algorithm.
        """
        self._router = router
        self.provider = provider
        self.embedding_model = embedding_model
        if consensus_config:
            self.consensus_config = consensus_config
        self.logger = logger.bind(router="chat")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """
        Set up FastAPI routes for the chat endpoint.
        """

        @self._router.post("/")
        async def chat(message: ChatMessage): # -> dict[str, str] | None:  # pyright: ignore [reportUnusedFunction]
            """
            Process a chat message through the CL pipeline.
            Returns an aggregated response after a number of iterations.
            """
            try:
                self.logger.debug("Received chat message", message=message.user_message)
                # We use the last 8 hours to make everything a bit faster
                from_ts = int(time.time()) - 3600 * 8
                to_ts = int(time.time())

                feed = FTSOFeed()
                wallet = WalletBalances()

                balances = wallet.get_balances("0x5a7338D940330109A2722140B7790fC4e286E54C")
                flr_usd_feed = feed.get_feed_analytics("FLR/USD", from_ts, to_ts)
                usdc_usd_feed = feed.get_feed_analytics("USDC/USD", from_ts, to_ts)

                initial_conversations: list[list[Message]] = []
                for i in range(len(self.consensus_config.models)):
                    initial_conversations.append(
                        [
                            {
                                "role": "system",
                                "content": self.consensus_config.models[i].system_prompt,
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
                    self.provider,
                    self.consensus_config,
                    initial_conversations,
                    self.embedding_model
                )

            except Exception as e:
                self.logger.exception("Chat processing failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e
            else:
                self.logger.info("Response generated", answer=answer)

                operation = extract_values(answer)
                return {"response": answer, "shapley_values": json.dumps(shapley_values), "operation": json.dumps(operation), "confidence_scores": confidence}

    @property
    def router(self) -> APIRouter:
        """Return the underlying FastAPI router with registered endpoints."""
        return self._router
