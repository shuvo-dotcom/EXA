"""Simple speech-to-text helpers.

This module provides small helpers to transcribe audio files using OpenAI's
Whisper model (when `OPENAI_API_KEY` is set) and a lightweight local fallback
using the `speech_recognition` package when available.

Functions:
 - transcribe_file(path, model) -> str
 - transcribe_bytes(data, filename_hint) -> str
 - local_fallback_transcribe(path) -> str

The Streamlit app `stt_app.py` (root) will use these helpers.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional


def transcribe_file(path: str, model: str = "whisper-1", timeout: Optional[int] = 60) -> str:
	"""Transcribe an audio file using OpenAI Whisper (via openai package).

	Requires the environment variable OPENAI_API_KEY to be set.

	Returns the transcribed text.
	"""
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("OPENAI_API_KEY not set. Set it to use OpenAI Whisper.")

	try:
		# openai >=1.0 uses the OpenAI client
		from openai import OpenAI
	except Exception as exc:  # pragma: no cover - environment dependent
		raise RuntimeError("openai package is not installed") from exc

	client = OpenAI(api_key=api_key)
	with open(path, "rb") as fh:
		# new client API: client.audio.transcriptions.create(...)
		resp = client.audio.transcriptions.create(model=model, file=fh)

	# response often exposes 'text'
	try:
		return resp.text
	except Exception:
		if isinstance(resp, dict):
			return resp.get("text", "")
		return str(resp)


def transcribe_bytes(data: bytes, filename_hint: str = "upload.wav", model: str = "whisper-1") -> str:
	"""Write bytes to a temp file and call transcribe_file.

	This is convenient for Streamlit file uploads where you get a BytesIO.
	"""
	suffix = os.path.splitext(filename_hint)[1] or ".wav"
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		tmp.write(data)
		tmp_path = tmp.name
	try:
		return transcribe_file(tmp_path, model=model)
	finally:
		try:
			os.remove(tmp_path)
		except Exception:
			pass


def local_fallback_transcribe(path: str) -> str:
	"""Fallback transcription using the `speech_recognition` package.

	This uses Google Web Speech API (internet required) via the
	`speech_recognition` library. It's intended as a low-dependency fallback
	when OPENAI_API_KEY is not set.
	"""
	try:
		import speech_recognition as sr
	except Exception as exc:  # pragma: no cover - optional dependency
		raise RuntimeError("speech_recognition is not installed; pip install SpeechRecognition") from exc

	r = sr.Recognizer()
	with sr.AudioFile(path) as source:
		audio = r.record(source)

	try:
		return r.recognize_google(audio)
	except sr.UnknownValueError:
		return ""
	except sr.RequestError as exc:
		raise RuntimeError("Speech recognition request failed") from exc

