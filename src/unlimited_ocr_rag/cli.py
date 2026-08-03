"""
Command-line interface

Examples
--------
    # Index files and ask one question, then exit:
    uocr-rag ask -f report.pdf -f table.png -q "extract all the tables"

    # Index files, save the vector index, and drop into an interactive REPL:
    uocr-rag chat -f report.pdf --save-index ./faiss_index

    # Reuse a saved index (no OCR re-run):
    uocr-rag chat --load-index ./faiss_index

    # Just OCR a file and print the extracted text:
    uocr-rag ocr -f scan.png
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Settings
from .ocr.factory import build_ocr_backend
from .rag import SimpleRAG


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-f", "--file", action="append", default=[], dest="files",
                   help="Input image/PDF (repeatable).")
    p.add_argument("--load-index", help="Load a saved FAISS index directory instead of running OCR.")
    p.add_argument("--save-index", help="Save the FAISS index to this directory after building.")


def _build_rag(args) -> SimpleRAG:
    rag = SimpleRAG(settings=Settings.from_env())
    if args.load_index:
        print(f"Loading index from {args.load_index} ...", file=sys.stderr)
        rag.load_index(args.load_index)
    else:
        if not args.files:
            sys.exit("No input files. Pass -f/--file or --load-index.")
        print(f"Indexing {len(args.files)} file(s) with OCR backend '{rag.ocr.name}' ...",
              file=sys.stderr)
        rag.build_index(args.files)
        if args.save_index:
            rag.save_index(args.save_index)
            print(f"Saved index to {args.save_index}", file=sys.stderr)
    return rag


def _print_answer(res: dict) -> None:
    print("\n" + res["answer"])
    if res.get("sources"):
        print("\nSources: " + ", ".join(res["sources"]))


def cmd_ocr(args) -> None:
    backend = build_ocr_backend(Settings.from_env())
    for path in args.files:
        print(f"# {path}  (backend: {backend.name})")
        print(backend.extract(path))
        print()


def cmd_ask(args) -> None:
    rag = _build_rag(args)
    _print_answer(rag.query(args.question))


def cmd_chat(args) -> None:
    rag = _build_rag(args)
    print("Interactive chat. Type a question, or 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("exit", "quit", ":q"):
            break
        if not q:
            continue
        _print_answer(rag.query(q))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uocr-rag", description="Unlimited OCR + RAG")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ocr = sub.add_parser("ocr", help="OCR files and print extracted text.")
    p_ocr.add_argument("-f", "--file", action="append", default=[], dest="files", required=True)
    p_ocr.set_defaults(func=cmd_ocr)

    p_ask = sub.add_parser("ask", help="Index files and answer a single question.")
    _add_common(p_ask)
    p_ask.add_argument("-q", "--question", required=True)
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Index files and chat interactively.")
    _add_common(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
