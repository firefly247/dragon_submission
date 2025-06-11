FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


# Install the requirements
COPY --chown=user:user requirements.txt .
RUN python -m pip install --user -r requirements.txt
COPY --chown=user:user dragon_baseline ./dragon_baseline

# Download the model, tokenizer and metrics
COPY --chown=user:user download_model.py .
# Download the model you want to use, e.g.:
RUN python download_model.py --model_name joeranbosma/dragon-roberta-base-mixed-domain

RUN python download_model.py --model_name joeranbosma/dragon-roberta-large-domain-specific
RUN python download_model.py --model_name joeranbosma/dragon-roberta-large-mixed-domain
RUN python download_model.py --model_name joeranbosma/dragon-longformer-large-domain-specific

COPY --chown=user:user download_metrics.py .
RUN python download_metrics.py

# Set the environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV HF_EVALUATE_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Copy the algorithm code
COPY --chown=user:user process.py .

ENTRYPOINT [ "python", "-m", "process" ]
