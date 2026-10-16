
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


def handler(event):
    data = event["input"]

    audio_url = data["audio"]
    model_name = data.get("model", "large-v3")
    min_speakers = data.get("min_speakers", 2)
    max_speakers = data.get("max_speakers", 2)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable missing")

    if not audio_url.startswith("https://"):
        raise ValueError("An HTTPS audio URL is required")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = str(Path(temp_dir) / "recording.mp3")

        # Download the original recording from R2
        with requests.get(
            audio_url, stream=True, timeout=300
        ) as response:
            response.raise_for_status()
            with open(audio_path, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # Transcribe with WhisperX
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language="en"
        )

        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=8)

        # Free memory before alignment
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        # Align words to the audio
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )

        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )

        del align_model
        if device == "cuda":
            torch.cuda.empty_cache()

        # Identify the speakers
        diarize_model = DiarizationPipeline(
            token=token,
            device=device
        )

        speaker_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        result = whisperx.assign_word_speakers(
            speaker_segments,
            result
        )

        # Return only the information needed
        segments = [
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "speaker": segment.get("speaker", "UNKNOWN"),
                "text": segment.get("text", "").strip()
            }
            for segment in result["segments"]
        ]

        return {"segments": segments}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
