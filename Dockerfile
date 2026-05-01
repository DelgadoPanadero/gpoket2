FROM python:3.10

# 1. Instalar herramientas del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    sudo \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

WORKDIR /home/gpoket2

# 2. Crear usuario no raíz (Por defecto 'devuser' si no se pasan ARGs)
ARG USERNAME=devuser
ARG USER_UID=1000
RUN groupadd --gid $USER_UID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_UID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/gpoket2 \
    && chown -R $USERNAME:$USERNAME /home/gpoket2 \
    && mkdir -p /home/$USERNAME/.ssh \
    && chmod 700 /home/$USERNAME/.ssh \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

EXPOSE 22

# 3. Instalar dependencias con UV (Usa --no-cache para optimizar tamaño)
RUN pip install uv
COPY ./pyproject.toml /home/gpoket2/pyproject.toml
RUN uv pip install --system --no-cache -r pyproject.toml
COPY ./ /home/gpoket2

# 4. Crear directorio para montar el volumen
ARG DATA_DIR=/workspace
RUN mkdir -p $DATA_DIR \
    && chmod 700 $DATA_DIR \
    && chown -R $USERNAME:$USERNAME $DATA_DIR

# 5. Cambiamos ENTRYPOINT al script inteligente
CMD ["/bin/bash", "/home/gpoket2/entrypoint_runpod.sh"]
