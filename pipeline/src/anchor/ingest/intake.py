"""Intake: turn "some video file" into a probed, content-addressed Episode.

This is the only door into the pipeline. Nothing downstream ever touches a
file that hasn't passed through :func:`ingest` — so nothing downstream ever
has to re-ask "what codec / how long / is this even decodable".

Probing shells out to ``ffprobe`` (ships with ffmpeg) rather than binding a
decoder library: it's the one dependency every robotics lab already has, and
it fails loudly on corrupt containers instead of half-decoding them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from anchor.ids import content_id
from anchor.schema.episode import Episode


class ProbeError(RuntimeError):
    """Raised when a file can't be probed — corrupt, truncated, or not video."""


@dataclass(frozen=True)
class VideoMeta:
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    n_frames: int | None


def probe(path: str | Path) -> VideoMeta:
    """Probe a video file with ffprobe. Raises :class:`ProbeError` on failure."""
    if shutil.which("ffprobe") is None:
        raise ProbeError("ffprobe not found on PATH — install ffmpeg")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        raise ProbeError(f"ffprobe timed out on {path}") from e
    if out.returncode != 0:
        raise ProbeError(f"ffprobe failed on {path}: {out.stderr.strip()[:300]}")

    data = json.loads(out.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ProbeError(f"no video stream in {path}")
    s = streams[0]

    num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 1) else 0.0
    nb = s.get("nb_frames")
    return VideoMeta(
        duration_s=float(data.get("format", {}).get("duration", 0.0)),
        fps=round(fps, 3),
        width=int(s.get("width", 0)),
        height=int(s.get("height", 0)),
        codec=s.get("codec_name", "unknown"),
        n_frames=int(nb) if nb and nb.isdigit() else None,
    )


def ingest(path: str | Path, source_uri: str | None = None) -> Episode:
    """Probe + content-address a local file into an :class:`Episode` record."""
    path = Path(path)
    meta = probe(path)
    ep = Episode(
        content_id=content_id(path),
        source_uri=source_uri or path.resolve().as_uri(),
        duration_s=meta.duration_s,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
    )
    return ep
