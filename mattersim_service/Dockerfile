# Use official NVIDIA PyTorch container which includes Python 3.12, PyTorch, CUDA, and cuDNN pre-installed.
FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /app

# Create a constraints file to lock PyTorch, torchvision, and NumPy to their pre-installed 
# versions in the container (preventing PyPI downloads and NumPy 2.x upgrade incompatibility).
RUN echo "torch==2.6.0a0+df5bbc09d1.nv24.12" > /tmp/constraints.txt && \
    echo "torchvision==0.20.0a0" >> /tmp/constraints.txt && \
    echo "numpy<2.0.0" >> /tmp/constraints.txt

# Install uv for fast, reliable, and parallel dependency resolution
RUN pip install --no-cache-dir uv

# Install Mattersim < 1.2.0 (NumPy 1.x compatible) and server dependencies.
# We bypass NVIDIA's slow mirrors by forcing the official PyPI simple index.
RUN PIP_CONFIG_FILE=/dev/null uv pip install --system --break-system-packages --index-url https://pypi.org/simple -c /tmp/constraints.txt "mattersim<1.2.0" fastapi uvicorn ase requests

# Copy the server script
COPY server.py /app/server.py

# Expose the API port
EXPOSE 8502

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8502"]
