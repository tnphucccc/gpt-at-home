# GPTAtHome

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

A character-level language model implementation using PyTorch. Generates Shakespeare-style text using Bigram and GPT model architecture.

## Features

- Character-level language modeling
- Two model architectures: Bigram and GPT
- Shakespeare text generation
- Interactive text generation interface
- CUDA support for GPU acceleration

## Requirements

- Python 3.12+ (with `venv`; on Ubuntu/WSL: `sudo apt install python3.12-venv`)
- [Bun](https://bun.sh) (or Node.js) for the web app
- NVIDIA GPU optional — the model runs fine on CPU

## Run locally

The project has two parts: a FastAPI server that serves the model (`server/`) and a React chat UI (`app/`). Run each in its own terminal.

**1. Server** (http://localhost:8000)
```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
fastapi dev main.py
```
On first start the trained weights (~50 MB) are downloaded from the GitHub release into `server/core/src/weight/model.pth`.
API docs are at http://localhost:8000/docs.

**2. Web app** (http://localhost:5173)
```bash
cd app
bun install
bun run dev
```
The app calls `http://localhost:8000` by default; set `VITE_API_URL` to point it elsewhere.

## Training the model

```bash
cd server/core
python src/train.py
```
This trains on `data/input.txt` (Tiny Shakespeare) and saves the best checkpoint to `server/core/checkpoints/model.pth`.
To serve your own checkpoint, copy it to `server/core/src/weight/model.pth`.

## Tests

```bash
server/.venv/bin/python -m pytest tests
```

## Project Structure

```bash
GPTAtHome/
├── app/                      # React + Vite chat UI
│   └── src/
├── server/
│   ├── main.py               # FastAPI app (POST /generate)
│   ├── api/context.py        # Request model, calls the runtime model
│   └── core/
│       ├── data/input.txt    # Training data (Tiny Shakespeare)
│       └── src/
│           ├── models/       # bigram.py, gpt.py
│           ├── utils/data_processor.py
│           ├── train.py      # Training script
│           └── generate.py   # Loads/downloads weights, generates text
└── tests/                    # Unit tests
```
