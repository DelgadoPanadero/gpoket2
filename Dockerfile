FROM python:3.10

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    sudo \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace
RUN pip install uv
COPY ./pyproject.toml /workspace/pyproject.toml
RUN uv pip install -r pyproject.toml

COPY ./ /workspace

# Create non-root user
ARG USERNAME
ARG USER_UID
RUN groupadd --gid $USER_UID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_UID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace

USER $USERNAME
