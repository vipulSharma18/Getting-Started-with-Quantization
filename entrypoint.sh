#!/bin/bash
set -e

# Ensure libcuda.so exists
if [ ! -f /lib/x86_64-linux-gnu/libcuda.so ] && [ -f /lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    echo "Creating libcuda.so symlink..."
    ln -s /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so
    ldconfig
fi

echo "starting HF download."

source /app/.venv/bin/activate
hf download --repo-type model unsloth/Meta-Llama-3.1-8B-Instruct

echo "download complete."

echo "setting up gemlite env"

uv sync --locked --group gemlite

echo "entrypoint finished"

exec "$@"