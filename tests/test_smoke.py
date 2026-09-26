"""Smoke tests for CampusMate."""
from __future__ import annotations

import subprocess
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CampusMateSmokeTests(unittest.TestCase):
    """End-to-end smoke checks for dataset, training, and API."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._backup_dir = Path(tempfile.mkdtemp(prefix="campusmate-artifacts-"))
        if (ROOT / "artifacts").exists():
            shutil.copytree(ROOT / "artifacts", cls._backup_dir / "artifacts", dirs_exist_ok=True)
        subprocess.run([sys.executable, "data/build_dataset.py"], cwd=ROOT, check=True)
        from training.train import train_model
        train_model("hybrid", epochs=2, patience=2, subset_per_class=8, save_main=True)

    @classmethod
    def tearDownClass(cls) -> None:
        backup = getattr(cls, "_backup_dir", None)
        if backup and (backup / "artifacts").exists():
            shutil.rmtree(ROOT / "artifacts", ignore_errors=True)
            shutil.copytree(backup / "artifacts", ROOT / "artifacts")
            shutil.rmtree(backup, ignore_errors=True)

    def test_dataset_builds(self) -> None:
        dataset = ROOT / "data" / "dataset.csv"
        self.assertTrue(dataset.exists())
        self.assertGreater(dataset.stat().st_size, 1000)

    def test_api_chat_and_fallback(self) -> None:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        ok = client.post("/api/chat", json={"text": "what time does the library close"})
        self.assertEqual(ok.status_code, 200)
        payload = ok.json()
        self.assertIn("intent", payload)
        self.assertIn("response", payload)
        gibberish = client.post("/api/chat", json={"text": "zzzx qwerp flindle blorptastic"})
        self.assertEqual(gibberish.status_code, 200)
        self.assertTrue(gibberish.json()["fallback"])


if __name__ == "__main__":
    unittest.main()
