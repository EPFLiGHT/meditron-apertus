ARG CUDA_VERSION="12.8.0"
ARG UBUNTU_VERSION="24.04"
ARG MAX_JOBS=3

FROM nvidia/cuda:$CUDA_VERSION-cudnn-devel-ubuntu$UBUNTU_VERSION AS base-builder
ENV PATH="/root/miniconda3/bin:${PATH}"

ARG PYTHON_VERSION="3.10"
# ARG PYTORCH_VERSION="2.1.2"
ARG PYTORCH_VERSION="2.9.0"
ARG CUDA="128"
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0+PTX"

ENV PYTHON_VERSION=$PYTHON_VERSION
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y --allow-change-held-packages vim curl nano libnccl2 libnccl-dev rsync s3fs && \
    rm -rf /var/cache/apt/archives && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget git build-essential ninja-build git-lfs libaio-dev pkg-config \
        ibverbs-providers ibverbs-utils infiniband-diags  \
        librdmacm-dev librdmacm1 rdmacm-utils slurm-wlm \
        cmake \
    && rm -rf /var/cache/apt/archives \
    && rm -rf /var/lib/apt/lists/* \
    && wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-aarch64.sh -b \
    && rm -f Miniconda3-latest-Linux-aarch64.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

ENV PATH="/root/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}"

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip && pip3 install -U packaging==23.2 setuptools==75.8.0 wheel psutil && \
    python3 -m pip install --no-cache-dir -U torch==${PYTORCH_VERSION}+cu${CUDA} torchvision --extra-index-url https://download.pytorch.org/whl/cu$CUDA && \
    python3 -m pip cache purge

RUN if [ "$CUDA" != "130" ] ; then \
        CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE python3 -m pip install --no-cache-dir "causal_conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.4"; \
        python3 -m pip install --no-cache-dir "mamba_ssm @ git+https://github.com/state-spaces/mamba.git@main"; \
        python3 -m pip cache purge; \
    fi

RUN git lfs install --skip-repo && \
    pip3 install awscli && \
    # The base image ships with `pydantic==1.8.2` which is not working
    pip3 install -U --no-cache-dir pydantic==1.10.10 && \
    pip3 cache purge

RUN wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.4/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_aarch64.whl; \
    pip3 install --no-cache-dir flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_aarch64.whl; \
    rm flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_aarch64.whl
    
WORKDIR /workspace


RUN git clone --depth=1 https://github.com/axolotl-ai-cloud/axolotl.git --branch v0.13.0
RUN git clone --depth=1 https://github.com/deepspeedai/DeepSpeed-Kernels.git

WORKDIR /workspace/DeepSpeed-Kernels

RUN pip install -v --no-build-isolation .

WORKDIR /workspace/axolotl

# If AXOLOTL_EXTRAS is set, append it in brackets
RUN if [ "$AXOLOTL_EXTRAS" != "" ] ; then \
        pip install --no-build-isolation .[deepspeed,flash-attn,$AXOLOTL_EXTRAS] $AXOLOTL_ARGS; \
    else \
        pip install --no-build-isolation .[deepspeed,flash-attn] $AXOLOTL_ARGS; \
    fi && \
    python scripts/unsloth_install.py | sh && \
    python scripts/cutcrossentropy_install.py | sh && \
    pip install pytest && \
    pip cache purge

# Fix so that git fetch/pull from remote works with shallow clone
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && \
    git config --get remote.origin.fetch && \
    git config --global credential.helper store
RUN cp .axolotl-complete.bash /root/.axolotl-complete.bash && \
    chmod +x /root/.axolotl-complete.bash && \
    echo 'source /root/.axolotl-complete.bash' >> ~/.bashrc

WORKDIR /workspace
RUN rm -rf /workspace/DeepSpeed-Kernels && \
    rm -rf /workspace/axolotl

########################################################################

# Copy the requirements.txt file to the working directory
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt && pip cache purge
WORKDIR /

# Final cleanup
RUN rm -rf /workspace
