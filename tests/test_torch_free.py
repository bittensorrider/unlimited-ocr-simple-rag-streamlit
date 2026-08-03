"""Regression guard: the default install path must stay torch-free.

The project defaults to fastembed (ONNX) embeddings specifically so it installs
and runs where PyTorch is unavailable or too old (e.g. Intel macOS, whose torch
ceiling is 2.2.2). Importing the package or building the pipeline with the
default config must therefore not import `torch`.

Run in a subprocess so a torch import triggered elsewhere in the test session
can't mask a regression here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_package_import_does_not_pull_torch():
    code = textwrap.dedent(
        """
        import sys
        import unlimited_ocr_rag
        from unlimited_ocr_rag import SimpleRAG, Settings
        assert Settings().embedding_provider == "fastembed"
        assert "torch" not in sys.modules, "importing the package pulled torch"
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
