from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from anchor.proof import _sha256


class KhiProofPrimitiveTest(unittest.TestCase):
    def test_hash_changes_when_evidence_is_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.txt"
            artifact.write_text("human-video-derived", encoding="utf-8")
            original = _sha256(artifact)
            artifact.write_text("pre-made-motion-substitution", encoding="utf-8")
            self.assertNotEqual(original, _sha256(artifact))


if __name__ == "__main__":
    unittest.main()
