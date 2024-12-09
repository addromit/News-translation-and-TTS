# Stage 1: Build Dependencies
FROM python:3.8-slim AS build-stage

# Combine RUN commands to reduce image layers
RUN apt-get update && \
    apt-get install -y git build-essential libsndfile1 libopenblas-dev liblapack-dev && \
    apt-get clean && \
    pip install --no-cache-dir Cython==0.29.21 numpy

# Combine repository cloning to reduce layers
RUN git clone https://github.com/VarunGumma/indic_nlp_library.git /indic_nlp_library && \
    git clone https://github.com/VarunGumma/IndicTransToolkit.git /IndicTransToolkit && \
    git clone https://github.com/jaywalnut310/vits.git /vits

# Build monotonic_align extensions
WORKDIR /vits/monotonic_align
RUN python3 setup.py build_ext --inplace

# Stage 2: Final Runtime Image
FROM python:3.8-slim

# Minimize system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libsndfile1 && \
    apt-get clean && \
    apt-get install -y wget tar &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only necessary directories
COPY --from=build-stage /indic_nlp_library ./indic_nlp_library
COPY --from=build-stage /IndicTransToolkit ./IndicTransToolkit
COPY --from=build-stage /vits ./vits

# Set environment variables more explicitly
ENV PYTHONPATH=/app/vits:/app/IndicTransToolkit:/app/indic_nlp_library \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies in one layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Improve model downloading with error handling
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('ai4bharat/indictrans2-en-indic-1B', trust_remote_code=True); \
    AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-en-indic-1B', trust_remote_code=True)"
# Dynamic device detection with fallback
ENV DEVICE=cpu
RUN python3 -c "\
import torch; \
device = 'cuda' if torch.cuda.is_available() else 'cpu'; \
print(f'Using device: {device}'); \
import os; \
os.environ['DEVICE'] = device"
COPY . .


RUN pip install librosa numpy pandas && \
    pip install --upgrade numba llvmlite pydub
RUN python3 -c "import nltk; nltk.download('punkt_tab')"
# RUN python3 -c "os.environ['NUMBA_DISABLE_CACHING'] = '1'"
# Disable Numba debug and caching messages
ENV NUMBA_DISABLE_CACHING=1
ENV NUMBA_OPT=1
ENV NUMBA_DEBUGINFO=0
# Suppress Numba debug and verbose logs
ENV NUMBA_LOG_LEVEL=ERROR

# Suppress Transformers and other libraries' logs
ENV TRANSFORMERS_VERBOSITY=error
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Suppress Python warnings
ENV PYTHONWARNINGS="ignore"

EXPOSE 8000
#uvicorn main:app --host 0.0.0.0 --port 8000 --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]