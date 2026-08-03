# unlimited-ocr-simple-rag-streamlit

> Implementation of Unlimited OCR + simple RAG + Streamlit app based on the [Medium Post](https://medium.com/data-science-collective/unlimited-ocr-rag-revolutionize-complex-data-extraction-open-source-f4199677ee55)

![Unlimited OCR](screenshot.png)

A faithful, runnable implementation of the article **"Unlimited OCR + RAG: Revolutionize Complex Data Extraction (Open-Source)"** by Gao Dalie. It pairs Baidu's [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) model — a 3B-parameter (≈500M active) vision-language OCR that parses an entire multi-page document in a _single_ forward pass — with a LangChain RAG stack (FAISS + BGE embeddings + GPT-4o).

The article's `SimpleRAG` class is reproduced exactly, then wrapped in a small, testable package with a CPU fallback, a CLI, and a Streamlit chat UI so it runs on any machine.

## Why Unlimited-OCR

Conventional OCR pipelines process a PDF page-by-page, then stitch the pieces back together — tables split across pages, headings go missing, reading order breaks. Unlimited-OCR keeps the whole document in context but slides a fixed window over the _generated_ text ("soft forgetting"), so the KV cache stays flat and long documents are read start-to-finish in one pass. Less preprocessing, less stitching, more consistent text handed to RAG.

## How it works

```
files (img/PDF) ──▶ OCR backend ──▶ text (Markdown)
                    │  Unlimited-OCR (GPU)  or  RapidOCR fallback (CPU)
                    ▼
             LangChain Documents (source kept as metadata)
                    ▼
   RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
                    ▼
        FAISS + BGE embeddings  ──▶  RetrievalQA (top-k=3, GPT-4o)
                    ▼
                 answer + sources
```

## Install

```bash
cd works
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # then add your OPENAI_API_KEY
```

Optional extras:

```bash
pip install -e ".[fallback]" # RapidOCR CPU OCR engine (also in requirements.txt)
pip install -e ".[ui]"       # Streamlit chat UI
pip install -e ".[dev]"      # pytest + ruff
pip install -e ".[gpu]"      # real Unlimited-OCR — needs an NVIDIA GPU + CUDA
pip install -e ".[hf]"       # article-faithful HuggingFace/BGE embeddings (pulls PyTorch)
```

The core install is **torch-free by design**: embeddings default to `fastembed`
(BGE via ONNX Runtime), so nothing pulls PyTorch. This keeps installs clean on
machines without a modern torch — notably **Intel macOS**, whose torch ceiling is
`2.2.2`, which is incompatible with current `transformers`/`numpy 2`. Only the
`gpu` (real model) and `hf` (sentence-transformers) extras bring in PyTorch.

> **Hit a torch / NumPy 2 / "transformers requires torch>=2.4" error?** You have an
> older env that pulled `sentence-transformers` + `torch`. Clean it out:
> ```bash
> pip uninstall -y sentence-transformers transformers torch torchvision
> pip install -e ".[fallback,ui,dev]"   # torch-free
> ```

## OCR backends

`OCR_BACKEND` (in `.env`) selects how text is extracted:

| Value            | Behavior                                                                 |
| ---------------- | ------------------------------------------------------------------------ |
| `auto` (default) | Real Unlimited-OCR if a CUDA GPU is present, otherwise the CPU fallback. |
| `unlimited`      | Force the real Baidu model. Requires GPU + `pip install -e ".[gpu]"`.    |
| `fallback`       | Force the RapidOCR CPU engine — lower quality, but runs anywhere.        |

The real backend calls the model exactly as the [model card](https://huggingface.co/baidu/Unlimited-OCR) documents: `infer_multi(...)` for PDFs (base config, `image_size=1024`, `ngram_window=1024`) and `infer(...)` for single images (gundam config, `base_size=1024`, `image_size=640`, `crop_mode=True`).

## Usage

### CLI

```bash
# OCR only — print extracted text
uocr-rag ocr -f scan.png

# Index files and ask one question
uocr-rag ask -f report.pdf -f table.png -q "extract all the tables"

# Index, save the vector index, and chat interactively
uocr-rag chat -f report.pdf --save-index ./faiss_index

# Reuse a saved index (skips OCR)
uocr-rag chat --load-index ./faiss_index
```

### Streamlit chat UI

```bash
streamlit run app.py
```

Upload images/PDFs in the sidebar, click **Index documents**, then chat.

## Configuration

All settings come from environment variables / `.env` (see `.env.example`): `OPENAI_API_KEY`, `LLM_PROVIDER` (`openai` default, `ollama` supported), `LLM_MODEL`, `EMBEDDING_PROVIDER` (`fastembed` default, `openai`, `huggingface`), `EMBEDDING_MODEL`, `OCR_BACKEND`, `OCR_MODEL`, `OCR_DPI`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_K`.

**Embedding providers:** `fastembed` (default) runs BGE via ONNX — local, free, torch-free. `openai` uses `text-embedding-3-small` (set `EMBEDDING_MODEL` accordingly; reuses `OPENAI_API_KEY`). `huggingface` is the article's exact sentence-transformers path — install `.[hf]` and use a machine with a working PyTorch.

## Tests

```bash
pytest
```

Tests inject a stub OCR backend, deterministic fake embeddings, and a fake chat model, so the full index→retrieve→answer flow runs offline — no GPU, no network, no API key. Backend-specific tests (PDF rendering, RapidOCR) skip automatically when their dependencies aren't installed.

## Notes & credits

- Model: [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (MIT), paper [arXiv:2606.23050](https://arxiv.org/abs/2606.23050).
- Article: [_Unlimited OCR + RAG: Revolutionize Complex Data Extraction_](https://medium.com/data-science-collective/unlimited-ocr-rag-revolutionize-complex-data-extraction-open-source-f4199677ee55) by Gao Dalie (高達烈).
- The fallback engine is for portability/testing; for production-grade tables, formulas, and reading order, run the real model on a GPU.

&copy; 2026 All rights reserved.
