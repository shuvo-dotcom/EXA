"""Record from the microphone and transcribe the audio.

Usage (PowerShell):
  python mic_stt.py --duration 8

Dependencies:
  pip install sounddevice soundfile

The script will use OPENAI_API_KEY if set (or read from api_keys/openai). If no
OpenAI key is found it will fall back to `speech_recognition` (Google Web Speech)
via the helpers in `src.tools.stt`.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import sys

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sys
import importlib.util
try:
    import sounddevice as sd
    import soundfile as sf
except Exception as exc:  # pragma: no cover
    # Detailed diagnostics for why audio imports failed
    print(f"Audio import error. Python executable: {sys.executable}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)
    print(f"sounddevice spec: {importlib.util.find_spec('sounddevice')}", file=sys.stderr)
    print(f"soundfile spec: {importlib.util.find_spec('soundfile')}", file=sys.stderr)
    print(f"Original audio import exception: {exc}", file=sys.stderr)
    sd = None  # type: ignore
    sf = None  # type: ignore

from src.tools import stt
import winsound
import shutil
import subprocess

def ensure_openai_key_from_repo() -> None:
    # Mirror the logic used in the Streamlit app: if OPENAI_API_KEY not set,
    # try to read from api_keys/openai or api_keys/openai.txt in the repo cwd.
    if os.getenv("OPENAI_API_KEY"):
        return
    candidate = os.path.join(os.getcwd(), "api_keys", "openai")
    candidate2 = os.path.join(os.getcwd(), "api_keys", "openai.txt")
    for p in (candidate, candidate2):
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    k = fh.read().strip()
                if k:
                    os.environ["OPENAI_API_KEY"] = k
                    return
        except Exception:
            continue

def record_to_wav(duration: float, samplerate: int = 16000) -> str:
    if sd is None or sf is None:
        raise RuntimeError("sounddevice and soundfile are required: pip install sounddevice soundfile")

    channels = 1
    print(f"Recording for {duration} seconds (samplerate={samplerate})...")
    try:
        data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
        sd.wait()
    except Exception as exc:
        raise RuntimeError(f"Failed to record audio: {exc}") from exc

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    play_chime(sound_path_input='speech_off')

    try:
        # soundfile expects float data or int16; our dtype is int16
        sf.write(tmp_path, data, samplerate, subtype='PCM_16')
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    return tmp_path


def main(duration, samplerate = 16000) -> int:
    ensure_openai_key_from_repo()
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    if use_openai:
        print("Using OpenAI Whisper (OPENAI_API_KEY found)")
    else:
        print("No OpenAI key found; using local fallback (SpeechRecognition) if available")

    try:
        wav_path = record_to_wav(duration, samplerate = samplerate)
    except Exception as exc:
        print(f"Error recording: {exc}", file=sys.stderr)
        return 2

    try:
        if use_openai:
            text = stt.transcribe_file(wav_path)
        else:
            text = stt.local_fallback_transcribe(wav_path)
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return 3

    print("\n--- Transcription ---\n")
    print(text or "(no text returned)")
    print("\n---------------------\n")

    return text



if __name__ == "__main__":
    main()
