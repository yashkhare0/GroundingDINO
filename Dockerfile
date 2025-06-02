FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

RUN conda update conda -y

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         python3-opencv \
         supervisor \
         ninja-build \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for all the subsequent Dockerfile instructions.
WORKDIR /opt/program

# Clone GroundingDINO and install it with pip
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO && \
    sed -i 's/torch.utils.cpp_extension.BuildExtension/torch.utils.cpp_extension.BuildExtension.with_options(no_cuda=True)/' setup.py && \
    pip install -e .

# Download model weights
RUN mkdir -p weights && \
    wget -q -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Install FastAPI and Uvicorn for the API service
RUN pip install fastapi uvicorn

# Copy necessary files
COPY docker_test.py /opt/program/docker_test.py
COPY api_service.py /opt/program/api_service.py
COPY supervisord.conf /opt/program/supervisord.conf

# Create directory for supervisor logs
RUN mkdir -p /var/log/supervisor

# Run supervisor which will start the API service
CMD ["supervisord", "-c", "/opt/program/supervisord.conf"]