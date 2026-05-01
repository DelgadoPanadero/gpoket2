#!/bin/bash

# Verificar si RunPod ha inyectado una clave SSH pública
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
fi

# Iniciar el demonio de SSH en primer plano
exec /usr/sbin/sshd -D
