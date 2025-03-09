# implement code for adding an agent to the system (to consensus_config)

import structlog
from fastapi import APIRouter

from flare_ai_consensus.settings import ConsensusConfig, ModelConfig

logger = structlog.get_logger(__name__)
router = APIRouter()


# expose this function to the API
class ModelRouter:
    def __init__(
            self,
            router: APIRouter,
            consensus_config: ConsensusConfig | None = None,
    ) -> None:
        self._router = router
        if consensus_config:
            self.consensus_config = consensus_config
        self.logger = logger.bind(router="chat")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self._router.post("/add-agent")
        async def add_agent(model_name: str, public_key: str):
            model = ModelConfig(
                model_id=model_name,
                max_tokens=50,
                temperature=0.5,
                public_key=public_key,
            )

            self.consensus_config.models.append(model)
            return {"status": "Successfully added model"}

        @self._router.post("/list-agents")
        async def list_agents():
            return [{"model_id": model.model_id, "public_key": model.public_key} for model in self.consensus_config.models]

    @property
    def router(self) -> APIRouter:
        return self._router
