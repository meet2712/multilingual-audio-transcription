# Multilingual audio transcription

This is a command-line transcription tool for English, Hindi, and Gujarati audio. It accepts common audio files such as MP3 and WAV, prepares them with FFmpeg, and writes the transcript as UTF-8 JSON with timestamps for each segment.

The current version runs locally on the CPU. I kept it as a script so the audio-processing and transcription flow are easy to run and review without setting up a web service.

## What it does

- Accepts an audio file from the command line.
- Supports automatic language detection or an explicit language code.
- Uses specialist Whisper models for Hindi and Gujarati.
- Normalizes audio and converts it to mono, 16 kHz PCM before transcription.
- Splits long recordings into roughly five-minute chunks.
- Returns the full transcript and timestamped segments in JSON.
- Saves Hindi and Gujarati text in UTF-8 without escaping the characters.

## Language and model selection

| Language | Code | Model used in automatic model mode |
| --- | --- | --- |
| English | `en` | `large-v3` |
| Hindi | `hi` | `ukta-app/whisper-hindi-ct2` |
| Gujarati | `gu` | `ukta-app/whisper-gujarati-ct2` |

Use `--language auto` when the language is unknown. The script starts with the multilingual model and can route detected Hindi or Gujarati audio to the corresponding specialist model.

## Requirements

- Python 3.9 or newer
- FFmpeg available on `PATH`
- `faster-whisper` 1.2.1

The models are downloaded on first use, so the first run needs an internet connection.

## Setup

On macOS, install FFmpeg with Homebrew:

```bash
brew install ffmpeg
```

Create a virtual environment and install the tested `faster-whisper` version:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install faster-whisper==1.2.1
```

Check that Python and FFmpeg are available:

```bash
python3 --version
ffmpeg -version
```

## Usage

Let the script detect the language:

```bash
python3 transcribe.py "recording.mp3" --language auto -o "transcript.json"
```

Transcribe English audio:

```bash
python3 transcribe.py "english_recording.mp3" --language en -o "english_transcript.json"
```

Transcribe Hindi audio:

```bash
python3 transcribe.py "hindi_recording.mp3" --language hi -o "hindi_transcript.json"
```

Transcribe Gujarati audio:

```bash
python3 transcribe.py "gujarati_recording.mp3" --language gu -o "gujarati_transcript.json"
```

Paths containing spaces should stay inside quotes.

## Output

The output file contains the complete text, timestamps for each recognized segment, chunk details, the language used, and the model selected for the job. A shortened example is shown below.

```json
{
  "file": "hindi_recording.mp3",
  "model": "auto",
  "models_used": [
    "ukta-app/whisper-hindi-ct2"
  ],
  "audio_normalized": true,
  "requested_language": "hi",
  "languages": [
    "hi"
  ],
  "duration_seconds": 55.104,
  "text": "Recognized text appears here.",
  "segments": [
    {
      "start": 0.0,
      "end": 10.0,
      "text": "Recognized segment text appears here."
    }
  ],
  "warnings": []
}
```

All timestamps are measured in seconds from the beginning of the original recording.

## How the pipeline works

1. The script validates that the input file exists and is not empty.
2. FFmpeg extracts the first audio stream, normalizes it, and converts it to mono, 16 kHz PCM.
3. Long recordings are split into approximately five-minute temporary files.
4. Whisper transcribes each chunk and filters silence with VAD.
5. Chunk offsets are added to the segment timestamps so they still match the original recording.
6. The result is written as UTF-8 JSON. Temporary audio files are removed automatically.

## Useful options

| Option | Purpose |
| --- | --- |
| `--language auto` | Detect the spoken language |
| `--language en` | Transcribe English audio |
| `--language hi` | Transcribe Hindi audio |
| `--language gu` | Transcribe Gujarati audio |
| `--model MODEL_NAME` | Use a specific multilingual Whisper model |
| `--no-vad` | Disable silence filtering when speech is being missed |
| `--no-normalize` | Keep the original audio level |
| `-o FILE` | Save the result to a UTF-8 JSON file |

## Current scope

This repository contains the local transcription pipeline. It does not currently include an HTTP API, shared job queue, database, or cloud storage layer. Those would be separate additions if the script were turned into a multi-user service.
