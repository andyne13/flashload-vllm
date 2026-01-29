FROM vllm/vllm-openai:v0.10.2

# Install build dependencies and runtime libraries
RUN apt-get update && apt-get install -y \
    git cmake build-essential ninja-build \
    libgoogle-glog-dev libgflags-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone flashtensors, patch Python version support, supervisord, and disable process group destruction
RUN git clone https://github.com/leoheuler/flashtensors.git /tmp/flashtensors && \
    cd /tmp/flashtensors && \
    sed -i 's/"3.8" "3.9" "3.10" "3.11"/"3.8" "3.9" "3.10" "3.11" "3.12"/g' CMakeLists.txt && \
    sed -i 's/3.8;3.9;3.10;3.11/3.8;3.9;3.10;3.11;3.12/g' cmake/utils.cmake && \
    sed -i 's/%(ENV_USER)s/root/g' flashtensors/supervisord.conf && \
    sed -i 's/command=python -m/command=python3 -m/g' flashtensors/supervisord.conf && \
    pip install . && \
    rm -rf /tmp/flashtensors

# Patch flashtensors api.py to skip process group destruction (fixes vLLM compatibility)
RUN sed -i 's/dist.destroy_process_group()/pass  # Skipped: dist.destroy_process_group() breaks vLLM reload/g' \
    /usr/local/lib/python3.12/dist-packages/flashtensors/api.py && \
    echo "Patched flashtensors api.py to skip process group destruction"

# Install additional dependencies
RUN pip install fastapi uvicorn[standard] pydantic sse-starlette pillow httpx

# Copy the entrypoint script
COPY entrypoint.py /app/entrypoint.py

WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["python3", "/app/entrypoint.py"]
