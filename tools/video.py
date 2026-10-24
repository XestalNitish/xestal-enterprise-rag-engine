import os
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Dict, List

from langchain_core.tools import tool

try:
    from tools._safe import safe_local_path
except ImportError:  # running the file directly / flat layout
    from _safe import safe_local_path

ERR = "[ERROR]"
NO_RESULTS = "[NO_RESULTS]"
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".aac")

FFMPEG_HELP = (
    "ffmpeg not found. Fix with ONE of these: "
    "(1) add ffmpeg's 'bin' folder to PATH, then fully restart VS Code and the terminal; "
    "(2) set env var FFMPEG_PATH to the full path of ffmpeg.exe; "
    "(3) run: pip install imageio-ffmpeg (bundled ffmpeg, no PATH needed)."
)


def _find_ffmpeg() -> str:
    """Locate ffmpeg: FFMPEG_PATH env var, then PATH, then the imageio-ffmpeg bundled binary."""
    env = os.environ.get("FFMPEG_PATH")
    if env:
        if os.path.isfile(env):
            return env
        raise RuntimeError(f"FFMPEG_PATH is set but the file does not exist: {env}")
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError(FFMPEG_HELP)


def _load_audio(path: str):
    """Decode any audio/video file to 16 kHz mono float32 samples using our own ffmpeg.

    We pass this array to Whisper directly. If we passed a file path instead, Whisper would
    run a bare `ffmpeg` command itself and fail whenever ffmpeg is not on PATH.
    """
    import numpy as np

    cmd = [_find_ffmpeg(), "-nostdin", "-i", path, "-vn", "-f", "s16le",
           "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "ignore")
        if "does not contain any stream" in err or "matches no streams" in err:
            raise ValueError("No audio track found in this file.")
        last_lines = " | ".join([ln.strip() for ln in err.splitlines() if ln.strip()][-3:])
        raise RuntimeError("FFmpeg audio extraction failed: " + last_lines[-300:])
    audio = np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError("No audio track found in this file.")
    return audio


@lru_cache(maxsize=2)
def _get_model(model_size: str):
    """Load the Whisper model once and reuse it (loading on every call is very slow)."""
    import whisper  # raises ImportError if not installed
    return whisper.load_model(model_size)


def _fmt(sec: float) -> str:
    s = int(sec)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def load_transcript_segments(source: str, model_size: str = "base") -> List[Dict]:
    """Structured loader for ingestion: one dict per spoken segment.

    Returns [{"text": "...", "metadata": {"source", "start", "end"}}] with times in seconds,
    so answers can cite "at 03:21".
    """
    path = safe_local_path(source)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video/Audio file not found at path: {source}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in VIDEO_EXTS + AUDIO_EXTS:
        raise ValueError(f"Unsupported format '{ext}'. Use .mp4, .mkv, .mp3, .wav, etc.")

    audio = _load_audio(path)
    model = _get_model(model_size)
    result = model.transcribe(audio, verbose=False)

    segments = [
        {"text": s["text"].strip(),
         "metadata": {"source": source, "start": float(s["start"]), "end": float(s["end"])}}
        for s in result.get("segments", []) if s.get("text", "").strip()
    ]
    if not segments:
        full = result.get("text", "").strip()
        if full:
            segments = [{"text": full, "metadata": {"source": source, "start": 0.0, "end": None}}]
    return segments


@tool
def video_reader(source: str, model_size: str = "base", max_chars: int = 500000) -> str:
    """Robust video and audio reader for enterprise RAG.
    Extracts audio using FFmpeg and transcribes it using local OpenAI Whisper,
    preserving timestamp segments for precise citations.

    Args:
        source: Local path to a video/audio file (.mp4, .mkv, .mov, .mp3, .wav).
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                    'base' or 'small' is recommended for local execution balance.
        max_chars: Maximum characters to return.
    """
    try:
        segments = load_transcript_segments(source, model_size=model_size)
        if not segments:
            return f"{NO_RESULTS} No speech detected or transcribed in the video/audio."
        lines = []
        for seg in segments:
            m = seg["metadata"]
            if m["end"] is None:
                lines.append(f"[Full Transcript]\n{seg['text']}")
            else:
                lines.append(f"[{_fmt(m['start'])} - {_fmt(m['end'])}] {seg['text']}")
        header = f"Source: {source}\nType: Video/Audio Transcription (Whisper Model: {model_size})\n\n"
        return header + "\n".join(lines)[:max_chars]
    except ImportError:
        return f"{ERR} 'whisper' library is not installed. Run: pip install openai-whisper"
    except (PermissionError, FileNotFoundError, ValueError, RuntimeError) as e:
        return f"{ERR} {e}"
    except Exception as e:
        return f"{ERR} video_reader error: {type(e).__name__}: {str(e)[:150]}"

