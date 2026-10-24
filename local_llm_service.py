"""
local_llm_service.py — 100% Offline & Local X-Estal-4B Inference Service
========================================================================
Manages local execution of the downloaded X-Estal GGUF model via llama-server.
Thinking/reasoning tags are strictly disabled for crisp, direct responses.
Zero external API calls. Zero paid dependencies. 100% Private, Autonomous, and Free.
"""

from __future__ import annotations

import os
import sys
import time
import json
import re
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

import requests

logger = logging.getLogger("x_estal_llm_service")

_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8088
DEFAULT_HOST = "127.0.0.1"

_server_process: Optional[subprocess.Popen] = None


def strip_thinking(text: str) -> str:
    """Remove any thinking tags or internal monologue from output."""
    if not text:
        return ""
    # Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # Strip [THINKING]...[/THINKING] blocks
    cleaned = re.sub(r"\[THINKING\][\s\S]*?\[/THINKING\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def get_model_path() -> Path:
    """Return path to downloaded GGUF weights."""
    candidates = [
            _ROOT / "models" / "Spark-X2.5-4B-GGUF" / "Spark-X2.5-4B-Q4_K_M.gguf",
            Path("D:/project3/models/Spark-X2.5-4B-GGUF/Spark-X2.5-4B-Q4_K_M.gguf"),
            _ROOT / "models" / "Spark-X2.5-1.7B-Q4_K_M.gguf",
            Path("D:/project3/models/Spark-X2.5-1.7B-Q4_K_M.gguf"),
        ]
    for c in candidates:
        if c.exists():
            return c
    model_dir = _ROOT / "models" / "Spark-X2.5-4B-GGUF"
    if model_dir.exists():
        for f in model_dir.glob("*.gguf"):
            return f
    for search_dir in [_ROOT / "models", Path("D:/agentic ai/models"), Path("d:/local llm/models")]:
        if search_dir.exists():
            for f in search_dir.rglob("*.gguf"):
                return f
    return candidates[0]


def get_server_binary() -> Optional[Path]:
    """Find local llama-server executable binary."""
    candidates = [
        _ROOT / "llama_bin" / "llama-server.exe",
        _ROOT / "llama_bin" / "llama-server",
        _ROOT.parent / "local llm" / "llama_bin" / "llama-server.exe",
        Path("d:/local llm/llama_bin/llama-server.exe"),
        Path("D:/agentic ai/llama_bin/llama-server.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages" / "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe" / "llama-server.EXE",
    ]
    for c in candidates:
        if c.exists():
            return c
    for name in ["llama-server", "llama-server.exe"]:
        in_path = shutil.which(name)
        if in_path:
            return Path(in_path)
    return None


def is_local_capable() -> bool:
    """Check if local llama-server is currently active or capable of running locally."""
    return is_server_alive() or (get_server_binary() is not None and get_model_path() is not None)


def is_server_alive(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check if the local inference server is up and responsive."""
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=0.25)
        return r.status_code == 200
    except Exception:
        try:
            r = requests.get(f"http://{host}:{port}/v1/models", timeout=0.25)
            return r.status_code == 200
        except Exception:
            return False


def start_local_server(
    port: int = DEFAULT_PORT,
    n_ctx: int = 2048,
    n_threads: int = 6,
    wait_timeout: int = 15
) -> bool:
    """
    Start the local llama-server process in background if not already running.
    Optimized for high-throughput single-slot CPU inference with Flash Attention.
    """
    global _server_process

    if is_server_alive(DEFAULT_HOST, port):
        logger.info(f"Local X-Estal server already active on http://{DEFAULT_HOST}:{port}")
        return True

    exe = get_server_binary()
    model = get_model_path()

    if not exe or not exe.exists():
        logger.info(f"Local llama-server binary not found on this host (cloud/remote deployment active).")
        return False

    if not model or not model.exists():
        logger.info(f"Local GGUF model weights not found on this host (cloud/remote deployment active).")
        return False

    cmd = [
        str(exe),
        "-m", str(model),
        "--port", str(port),
        "--host", DEFAULT_HOST,
        "-c", str(n_ctx),
        "-t", str(n_threads),
        "-tb", str(n_threads),
        "-np", "1",                       # Single dedicated slot to prevent context fragmentation
        "-b", "1024",                     # Logical batch
        "-ub", "512",                     # Micro batch
        "-fa", "on",                      # Flash Attention
        "--prio", "2",                    # High process priority
        "--reasoning", "off",             # Disable thinking engine
        "--reasoning-budget", "0",        # Zero reasoning tokens
        "--no-reasoning-preserve",        # Do not preserve reasoning trace
        "-lv", "0",                       # Quiet logs
    ]

    logger.info(f"Starting local X-Estal inference server on port {port}...")
    try:
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000  # CREATE_NO_WINDOW

        _server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_timeout:
            if is_server_alive(DEFAULT_HOST, port):
                logger.info(f"✅ Local X-Estal server ready on http://{DEFAULT_HOST}:{port}")
                return True
            time.sleep(0.5)

        logger.warning(f"Server start timed out after {wait_timeout}s.")
        return False
    except Exception as e:
        logger.error(f"Failed to launch llama-server: {e}")
        return False


def query_local_model(
    messages: List[Dict[str, str]],
    max_tokens: int = 1000000,
    temperature: float = 0.2,
    port: int = DEFAULT_PORT
) -> Dict[str, Any]:
    """
    Direct synchronous completion query to the local X-Estal model.
    Thinking tags are stripped; returns only direct, clean answer.
    """
    if not is_server_alive(DEFAULT_HOST, port):
        started = start_local_server(port=port)
        if not started:
            raise RuntimeError(
                "Local X-Estal server is offline and could not be started automatically. "
                "Ensure model is present in models/."
            )

    url = f"http://{DEFAULT_HOST}:{port}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Local server returned HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})

    raw_content = msg.get("content", "")
    content = strip_thinking(raw_content)

    # In case content was completely inside think tags, fallback to cleaned version
    if not content:
        raw_reasoning = msg.get("reasoning_content", "")
        content = strip_thinking(raw_reasoning) or "Done."

    return {
        "content": content,
        "display": content,
        "usage": data.get("usage", {}),
    }


def stream_local_model(
    messages: List[Dict[str, str]],
    max_tokens: int = 1000000,
    temperature: float = 0.2,
    port: int = DEFAULT_PORT
) -> Generator[str, None, None]:
    """
    Stream clean response tokens directly from local X-Estal model.
    Filters out any thinking tags from the live stream.
    """
    if not is_server_alive(DEFAULT_HOST, port):
        start_local_server(port=port)

    url = f"http://{DEFAULT_HOST}:{port}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    in_thinking_block = False
    buffer = ""

    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_part = line_str[6:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_part)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if not token:
                        continue

                    # Filter out <think> blocks on the fly
                    buffer += token
                    if "<think>" in buffer.lower():
                        in_thinking_block = True
                    if in_thinking_block:
                        if "</think>" in buffer.lower():
                            # Exit thinking block
                            parts = buffer.lower().split("</think>", 1)
                            after = buffer[len(parts[0]) + 8:]
                            buffer = ""
                            in_thinking_block = False
                            if after:
                                yield after
                        continue

                    yield token
                    buffer = ""
                except Exception:
                    continue


def get_local_chat_llm(temperature: float = 0.2, port: int = DEFAULT_PORT, auto_start: bool = False, max_tokens: int = 150):
    """
    Return a LangChain ChatOpenAI instance connected to local X-Estal model.
    """
    if auto_start and not is_server_alive(DEFAULT_HOST, port):
        try:
            start_local_server(port=port, wait_timeout=15)
        except Exception:
            pass

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=f"http://{DEFAULT_HOST}:{port}/v1",
        api_key="x-estal-local",
        model_name="X-Estal-4B",
        temperature=temperature,
        request_timeout=180,
        max_tokens=max_tokens,
        max_retries=1,
    )
