FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

RUN apt-get update

RUN pip install \
    transformers \
    datasets \
    sentencepiece

WORKDIR /app

# RUN python -c "from transformers import AutoProcessor, AutoModel; AutoProcessor.from_pretrained('suno/bark-small'); AutoModel.from_pretrained('suno/bark-small')"
RUN python -c "from transformers import pipeline; pipeline('text-to-speech', 'microsoft/speecht5_tts')"

RUN pip install \
    fastapi \
    uvicorn \
    python-multipart

COPY . /app

ENTRYPOINT [ "uvicorn", "run:app", "--host", "0.0.0.0", "--port", "7860" ]
# ENTRYPOINT [ "/bin/bash" ]