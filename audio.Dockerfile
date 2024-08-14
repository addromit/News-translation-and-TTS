# Use an official Python runtime as a parent image
FROM python:3.9

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        gcc \
        git \
        curl \
        build-essential \
        ffmpeg \   
    && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

# Download and unzip en2indic.zip
RUN git clone https://github.com/AI4Bharat/IndicTrans2.git \
 && cd /app/IndicTrans2/huggingface_interface \
 && git clone https://github.com/VarunGumma/IndicTransTokenizer \
 && cd IndicTransTokenizer \
 && pip install --editable ./ \
 && cd ..

# Download TTS files
RUN curl -O https://dl.fbaipublicfiles.com/mms/tts/tam.tar.gz \
    && curl -O https://dl.fbaipublicfiles.com/mms/tts/tel.tar.gz \
    && curl -O https://dl.fbaipublicfiles.com/mms/tts/hin.tar.gz \
    && curl -O https://dl.fbaipublicfiles.com/mms/tts/guj.tar.gz \
    && curl -O https://dl.fbaipublicfiles.com/mms/tts/kan.tar.gz \
    && curl -O https://dl.fbaipublicfiles.com/mms/tts/mal.tar.gz \
    && curl -O https://dl.fbaipublicfiles.com/mms/tts/ben.tar.gz \
    && curl -O https://dl.fbaipublicfiles.com/mms/tts/pan.tar.gz \
    && curl -O https://dl.fbaipublicfiles.com/mms/tts/mar.tar.gz

# Extract TTS files
RUN mkdir -p data \
    && tar -xzf tam.tar.gz -C data/ \
    && tar -xzf tel.tar.gz -C data/ \
    && tar -xzf hin.tar.gz -C data/ \
    && tar -xzf guj.tar.gz -C data/ \
    && tar -xzf kan.tar.gz -C data/ \
    && tar -xzf mal.tar.gz -C data/ \
    && tar -xzf ben.tar.gz -C data/ \
    && tar -xzf pan.tar.gz -C data/ \
    && tar -xzf mar.tar.gz -C data/

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run uvicorn main:app --host 0.0.0.0 --port 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]











