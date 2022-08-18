FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir -p /workspace && chown algorithm:algorithm /workspace
RUN mkdir -p /opt/algorithm /workspace/inputs /workspace/outputs \
    && chown algorithm:algorithm /opt/algorithm /workspace/inputs /workspace/outputs

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"
ENV RESULTS_FOLDER="/opt/algorithm/nnunet/results"

RUN python -m pip install -U pip

# Copy nnU-Net results folder
COPY --chown=algorithm:algorithm nnunet/ /opt/algorithm/nnunet/

# Install algorithm requirements
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install -r requirements.txt

# Copy your own dependencies to the algorithm container folder
COPY --chown=algorithm:algorithm docker_inference_final.py /opt/algorithm/
COPY --chown=algorithm:algorithm custom_nnunet_predict.py /opt/algorithm/
COPY --chown=algorithm:algorithm model_restore.py /opt/algorithm/
COPY --chown=algorithm:algorithm PostProcessing.py /opt/algorithm/

# Copy predict.sh
COPY --chown=algorithm:algorithm predict.sh /opt/algorithm/
