# Stage 2: Build Backend
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS backend-builder
ADD . /flare-ai-defai
WORKDIR /flare-ai-defai
RUN uv sync --frozen

# Stage 3: Final Image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install supervisor and curl (removed nginx)
RUN apt-get update && apt-get install -y supervisor curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=backend-builder /flare-ai-defai/.venv ./.venv
COPY --from=backend-builder /flare-ai-defai/src ./src
COPY --from=backend-builder /flare-ai-defai/pyproject.toml .
COPY --from=backend-builder /flare-ai-defai/README.md .

# Setup supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Allow workload operator to override environment variables
LABEL "tee.launch_policy.allow_env_override"="GEMINI_API_KEY,GEMINI_MODEL,WEB3_PROVIDER_URL,WEB3_EXPLORER_URL,SIMULATE_ATTESTATION,OPEN_ROUTER_API_KEY,GEMINI_EMBEDDING_KEY"
LABEL "tee.launch_policy.log_redirect"="always"

EXPOSE 80

# Start supervisor (which will start the backend)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]