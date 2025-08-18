FROM python:3.10

WORKDIR /home

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/0.8.4/install.sh | sh

#COPY ./pyproject.toml /home

#COPY ./uv.lock /home

#RUN uv sync --locked --no-install-project

COPY ./ /home
