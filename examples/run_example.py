from __future__ import annotations

import sys

from unlimited_ocr_rag import SimpleRAG


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    file_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Summarize this document."

    rag = SimpleRAG()               # OCR backend auto-selected (GPU model or CPU fallback)
    rag.build_index([file_path])    # also accepts .pdf
    result = rag.query(question)

    print(result["answer"])
    if result["sources"]:
        print("\nSources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
