"""
audio_utils.py — wrap raw PCM bytes from Flutter into a valid WAV file
on disk so faster-whisper can decode it.
"""

import os
import wave
import tempfile


def save_wav(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Wrap raw PCM bytes (or a full WAV) in a proper WAV file on disk.
    Returns the path to the temp WAV file.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes is empty")

    print(f"[audio_util] RAW SIZE: {len(audio_bytes)}")
    pcm = audio_bytes

    # If Flutter accidentally sent a full WAV, strip its 44-byte RIFF header
    if pcm[:4] == b"RIFF":
        print("[audio_util] stripping RIFF header")
        pcm = pcm[44:]

    if len(pcm) < 64:
        raise ValueError(f"PCM too small: {len(pcm)}")

    tmp  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = tmp.name
    tmp.close()

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)

    size     = os.path.getsize(path)
    duration = len(pcm) / 2 / sample_rate
    print(f"[audio_util] WAV saved → {path}  ({size} bytes, {duration:.2f}s)")
    return path