# TTS Server

A simple web server that generates speech from text using Coqui TTS (based on Mozilla's TTS project).

## Features

- Generate speech from text in multiple languages
- Support for English and Polish languages
- Automatic cleanup of temporary files
- Docker support
- Streaming response for generated audio
- Compressed MP3 output (192kbps)
- Customizable output filename

## Requirements

- Docker
- Docker Compose (optional)

## Running with Docker

1. Build the Docker image:
```bash
docker build -t tts-server .
```

2. Run the container:
```bash
docker run -p 8000:8000 tts-server
```

## API Usage

### Generate Speech Endpoint

**POST** `/generate1`

Request body:
```json
{
    "text": "Hello, this is a test message.",
    "language": "en",  // optional, defaults to "en"
    "title": "My Speech"  // optional, used for the output filename
}
```

Example using curl:
```bash
curl -X POST http://localhost:8000/generate1 \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test message.", "language": "en", "title": "My Speech"}' \
  --output my_speech.mp3
```

The endpoint will return the generated audio file as an MP3 file with a filename based on the provided title. If no title is provided, a random filename will be generated.

## Supported Languages

- English (en) - using tacotron2-DDC model
- Polish (pl) - using mai_female/vits model

## Notes

- The server uses Coqui TTS for text-to-speech generation
- Temporary files are automatically cleaned up after generation
- The server supports streaming response for generated audio
- Audio is generated in MP3 format with 192kbps bitrate
- Filenames are automatically sanitized to remove invalid characters