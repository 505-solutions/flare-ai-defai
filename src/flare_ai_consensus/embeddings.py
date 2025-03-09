from google import genai
from google.genai import types


class EmbeddingModel:

    def __init__(
            self, api_key: str | None = None, base_url: str = "https://api.gemini.com/embedding"
    ) -> None:
        self.base_url = base_url.rstrip("/")  # Ensure no trailing slash
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)
        self.dim = 512

    async def get_embeddings(self, payload: str):
        return self.client.models.embed_content(
            model="text-embedding-004",
            contents=payload,
            config=types.EmbedContentConfig(output_dimensionality=self.dim),
        )

