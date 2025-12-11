# ♟️ Searchless Chess: Master-Level Chess Through Pure Neural Intuition

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.12-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-orange)](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TL;DR:** Neural chess engine achieving ~1960 ELO without search. 
> Vision Transformers prove 10x more parameter-efficient than ResNets. 
> Trained on 316M positions over 200+ hours on A100. All code, models, 
> and dataset openly available.

A neural network that plays chess without search algorithms—relying purely 
on learned intuition from millions of positions evaluated by Stockfish.

**Inspired by:** Google DeepMind's [*"Grandmaster-Level Chess Without Search"*](https://arxiv.org/html/2402.04494v1)

---

## 🎯 About

This project demonstrates that neural networks can develop chess intuition without performing any search. The model:

- **Evaluates positions** based on centipawn (cp) scores from deep Stockfish analysis
- **Predicts optimal moves** directly from FEN notation
- **Learns strategic patterns** from millions of positions

No minimax. No alpha-beta pruning. Pure neural intuition.

## 🔗 Quick Links

- 📦 [HuggingFace Dataset](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized) - 316M deduplicated positions
- 🤖 [Pre-trained Models](#models) - Download ready-to-use weights (coming soon)
- 📓 [Technical Blog Post](#) - Deep dive into methodology (coming soon)
- 🎮 [Play Against the Bot](#) - Interactive demo (coming soon)

## 🏆 Key Achievements

- ✅ **~1960 ELO without search** - pure value-based move selection
- ✅ **10x parameter efficiency** - ViT outperforms ResNet with ~90% fewer parameters
- ✅ **316M position dataset** - deduplicated, normalized, published on HuggingFace
- ✅ **Rigorous validation** - evaluated on 1200 Lichess puzzles across 12 difficulty tiers
- ✅ **200+ hours on NVIDIA A100** - self-funded training demonstrating individual-scale research

## 📊 Model Comparison

Model          | Params | Dataset   | Training | ELO  | ELO/1M | Search?
---------------|--------|-----------|----------|------|--------|--------
Stockfish      | ~15M   | ~5B       | -        | 3500 | 233    | Yes
DeepMind       | 270M   | ~15B      | XXX      | 2895 | 10.7   | No*
**ViT-large**  | 9.5M   | ~316M     | 72h      | 1960 | 206    | No
**ViT-small**  | 2.64M  | ~316M     | 24h      | 1817 | 688    | No

*DeepMind's model used action prediction (policy head) trained on 
move sequences, not explicit search. See [their paper](https://arxiv.org/html/2402.04494v1) for details.

### Efficiency Analysis

Working with:
- **28x fewer parameters** (9.5M vs DeepMind's 270M)
- **47x less training data** (316M vs 15B positions)
- **Individual compute budget** (>200h on NVIDIA A100)

The model achieved **67% of DeepMind's ELO** using **<4% of their resources**.

**Key insight:** Neural chess intuition scales efficiently at smaller scales, 
demonstrating that cutting-edge AI research is accessible beyond corporate labs.

---

## 🔬 Key Discoveries

### Vision Transformers vs ResNets

![Model Comparison Chart](docs/images/model_comparison.png "Performance vs Parameters: ViT achieves 10x parameter efficiency over ResNet")

*Figure 1: ELO performance vs model size. ViT achieves superior parameter efficiency.*

The most surprising finding: **Vision Transformers are dramatically more 
parameter-efficient than ResNets for chess position evaluation.**

**Key observations:**
- ViT-small (2.64M) achieved higher ELO thank ResNet-large (24M) → **~10x fewer parameters**
- ResNet plateaus: 12M→24M params = only +8 ELO improvement
- ViT scaling: continues improving, suggesting headroom for larger models
- ViT-small achieves **688 ELO per million parameters** vs ResNet-L ~133 and ResNet-XL ~70

**Why transformers excel:**
Chess positions require global reasoning—understanding how pieces across 
the entire board coordinate. ViT's self-attention mechanism naturally captures 
these long-range dependencies, while ResNet's local convolutions must stack 
many layers to approximate the same receptive field.

This aligns with recent research showing transformer superiority in tasks 
demanding holistic scene understanding.

---

## 📊 Dataset

### Original training data from **Lichess** via HuggingFace:
- [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
- ~784M of positions with Stockfish deep analysis

### Transformed, ready-to-Use Training Dataset (HuggingFace)

A fully processed, **deduplicated** version of the Lichess evaluation database is available for this project:

- **316,072,343** unique chess positions (FEN-based deduplication, keeping max `depth`)
- Stored in **Parquet**, split into **10 parts** (~32M positions each)
- Optimized for ML pipelines — fast loading, reduced size, unnecessary fields removed
- Licensed under **CC BY 4.0**

Example of usage:
```python
from datasets import load_dataset
dataset = load_dataset("mateuszgrzyb/lichess-stockfish-normalized", split="train")
```
Dataset page: [https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized)

**Processing:** This dataset is the output of `scripts/stage_2_deduplicate_data.py`, which performs global deduplication across all Stage 1 files, then splits the result into 10 parts for distribution.

---

## 🚀 Installation

### Download code
```bash
git clone https://github.com/mateuszgrzyb-pl/searchless-chess.git
cd searchless-chess
```

### Create venv with Poetry... (recommended)
```bash
poetry install
poetry shell
```

### ...or install requirements with pip.
```bash
pip install -r requirements.txt
```

---

## 📚 References

- [Google DeepMind: Grandmaster-Level Chess Without Search](https://arxiv.org/html/2402.04494v1)
- [DeepMind GitHub Repository](https://github.com/google-deepmind/searchless_chess)
- [Lichess Dataset on HuggingFace](https://huggingface.co/datasets/Lichess/chess-position-evaluations)

---

## 📄 License

MIT License

---

## 📧 Contact

* LinkedIn: [Mateusz Grzyb](https://www.linkedin.com/in/mateusz--grzyb/)
* Blog PL: [MateuszGrzyb.pl](https://mateuszgrzyb.pl)