FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for TTS and audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS
RUN wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz && \
    tar xvf piper_linux_x86_64.tar.gz && \
    mv piper/piper /usr/local/bin/ && \
    cp -r piper/espeak-ng-data /usr/share/ && \
    cp -r piper/* /usr/lib/ && \
    rm -rf piper piper_linux_x86_64.tar.gz

# Copy requirements first to leverage Docker cache
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create tmp directory for audio files
RUN mkdir -p /app/tmp

# Copy initialization script and run it to download TTS models
COPY init.py .
RUN python init.py

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]