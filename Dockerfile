# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12.12
FROM python:${PYTHON_VERSION}-slim as base

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install Ollama (before switching users)
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose port for Streamlit
EXPOSE 8000

# Run Streamlit and pull models if not already present
CMD ollama pull llama3.2 && ollama pull mistral && \
    streamlit run codes/streamlit_board_game_qa.py --server.port 8000 --server.address 0.0.0.0