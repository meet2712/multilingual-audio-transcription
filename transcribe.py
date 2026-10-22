"""A small Whisper script for English, Hindi, and Gujarati recordings."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from faster_whisper import WhisperModel


SPECIALIST_MODELS = {
    "gu": "ukta-app/whisper-gujarati-ct2",
    "hi": "ukta-app/whisper-hindi-ct2",
}


def get_ffmpeg_executable():
    """Use FFmpeg on PATH, or the project-installed bundled executable."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "FFmpeg is missing. Install it and add it to PATH, or install "
            "the project's requirements.txt."
        ) from exc


def get_audio_path(file_name):
    path = Path(file_name).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError("The audio file is empty.")
    return path


def normalize_language(value):
    value = value.strip().lower()
    aliases = {"english": "en", "hindi": "hi", "gujarati": "gu"}
    return None if value == "auto" else aliases.get(value, value)


def transcribe_audio(
    audio_path, language=None, model_name="auto", use_vad=True, normalize_audio=True
):
    ffmpeg = get_ffmpeg_executable()

    transcript = []
    chunks_info = []
    languages = set()
    models = {}
    models_used = set()
    offset = 0.0

    def load_model(name):
        if name not in models:
            print(f"Loading {name}; the first run downloads the model...", file=sys.stderr)
            model = WhisperModel(name, device="cpu", compute_type="int8")
            if len(model.supported_languages) <= 1:
                raise ValueError("Use a multilingual model, such as large-v3, rather than an .en model.")
            models[name] = model
        return models[name]

    def recognize(model, chunk, selected_language):
        if selected_language is not None and selected_language not in model.supported_languages:
            raise ValueError(f"Unsupported language code: {selected_language}")
        return model.transcribe(
            str(chunk), language=selected_language, task="transcribe",
            log_progress=True, beam_size=5, temperature=0.0,
            # Gujarati was getting cut off in longer windows, so keep these short.
            chunk_length=5 if selected_language == "gu" else 10,
            condition_on_previous_text=False, vad_filter=use_vad,
            language_detection_segments=3,
        )

    with TemporaryDirectory() as folder:
        command = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-xerror", "-i", str(audio_path), "-map", "0:a:0", "-vn",
        ]
        if normalize_audio:
            command += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
        command += [
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "segment", "-segment_time", "300",
            "-reset_timestamps", "1",
            str(Path(folder) / "chunk_%06d.wav"),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"Could not process the audio: {exc.stderr.strip()}") from exc

        chunks = []
        for path in sorted(Path(folder).glob("chunk_*.wav")):
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            if duration > 0:
                chunks.append((path, duration))
        if not chunks:
            raise ValueError("The file contains no audio samples.")

        for number, (chunk, duration) in enumerate(chunks, start=1):
            print(f"Transcribing chunk {number}/{len(chunks)}...", file=sys.stderr)
            selected_model = model_name
            if model_name == "auto":
                selected_model = SPECIALIST_MODELS.get(language, "large-v3")
            model = load_model(selected_model)
            segments, info = recognize(model, chunk, language)
            probability = info.language_probability if language is None else None

            if language is None and info.duration_after_vad > 0:
                detected_language = info.language
                routed_model = (
                    SPECIALIST_MODELS.get(detected_language, selected_model)
                    if model_name == "auto" else selected_model
                )
                # Detect the language first, then switch models before decoding the text.
                if routed_model != selected_model or detected_language == "gu":
                    segments.close()
                    selected_model = routed_model
                    segments, info = recognize(load_model(selected_model), chunk, detected_language)

            models_used.add(selected_model)

            found_text = False
            for segment in segments:
                text = " ".join(segment.text.split())
                if text:
                    found_text = True
                    # Whisper can include its padding in the last timestamp.
                    start = min(duration, max(0.0, segment.start))
                    end = min(duration, max(start, segment.end))
                    transcript.append({
                        "start": round(offset + start, 3),
                        "end": round(offset + end, 3),
                        "text": text,
                    })

            if found_text:
                languages.add(info.language)
            chunks_info.append({
                "start": round(offset, 3),
                "end": round(offset + duration, 3),
                "language": info.language if found_text else None,
                "model": selected_model,
                "language_probability": (
                    round(probability, 4)
                    if language is None and found_text else None
                ),
                "speech_seconds": round(info.duration_after_vad, 3) if use_vad else None,
            })
            if found_text:
                action = "Detected" if language is None else "Using"
                print(f"{action} language: {info.language}", file=sys.stderr)

            # Silent chunks still count toward the timestamps that follow.
            offset += duration

    warnings = []
    if not transcript:
        warnings.append(
            "No text was recognized. If the recording contains speech, try its "
            "language code (for example --language gu or --language hi), or --no-vad."
        )
    if any("\ufffd" in segment["text"] for segment in transcript):
        warnings.append("Some characters could not be decoded cleanly. Review the transcript.")

    return {
        "file": audio_path.name,
        "model": model_name,
        "models_used": sorted(models_used),
        "audio_normalized": normalize_audio,
        "requested_language": language or "auto",
        "languages": sorted(languages),
        "duration_seconds": round(offset, 3),
        "text": " ".join(segment["text"] for segment in transcript),
        "segments": transcript,
        "chunks": chunks_info,
        "warnings": warnings,
    }


def save_result(result, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        # Keep the non-English text intact when this is run from Windows too.
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Transcribe multilingual audio to UTF-8 JSON.")
    parser.add_argument("audio_file", help="Path to the audio recording")
    parser.add_argument("--language", default="auto", help="auto, gu, hi, en, or another supported code")
    parser.add_argument("--model", default="auto", help="auto selects Gujarati/Hindi specialists or large-v3; alternatively name a multilingual model")
    parser.add_argument("--no-vad", action="store_true", help="Disable silence filtering if speech is being missed")
    parser.add_argument("--no-normalize", action="store_true", help="Keep the original audio level")
    parser.add_argument("-o", "--output", help="Save the result as a UTF-8 JSON file")
    args = parser.parse_args()

    try:
        audio_path = get_audio_path(args.audio_file)
        output_path = Path(args.output).expanduser().resolve() if args.output else None
        if output_path == audio_path:
            raise ValueError("The output file must be different from the input audio file.")
        result = transcribe_audio(
            audio_path, normalize_language(args.language), args.model,
            use_vad=not args.no_vad, normalize_audio=not args.no_normalize,
        )
        for warning in result["warnings"]:
            print(f"Warning: {warning}", file=sys.stderr)
        if output_path is not None:
            save_result(result, output_path)
            print(f"Saved transcript: {output_path}", file=sys.stderr)
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        parser.exit(130, "Transcription cancelled.\n")
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
