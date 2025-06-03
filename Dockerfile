FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

RUN conda update conda -y

# Install libraries
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         ninja-build \
         supervisor \
         ca-certificates \
         libsm6 \
         libxext6 \
         libxrender-dev \
         libglib2.0-0 \
         libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install opencv via pip instead of apt to avoid dependency issues
RUN pip install opencv-python-headless

# Set the working directory for all the subsequent Dockerfile instructions.
WORKDIR /opt/program

# Copy the GroundingDINO files into the container
COPY . /opt/program/GroundingDINO/

RUN mkdir weights ; cd weights ; wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth ; cd ..

# Verify CUDA setup
RUN ls -la /usr/local/cuda/bin/nvcc && \
    echo "CUDA_HOME: $CUDA_HOME" && \
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('CUDNN version:', torch.backends.cudnn.version())"

# Install GroundingDINO
RUN cd GroundingDINO/ && python -m pip install -e .

# Install FastAPI and other dependencies for the API service
RUN pip install fastapi uvicorn pydantic requests

# Create log directory for supervisor
RUN mkdir -p /var/log/supervisor

# Start with supervisord
CMD ["/usr/bin/supervisord", "-c", "/opt/program/GroundingDINO/supervisord.conf"]
