# ♟️ Searchless Chess: Master-Level Chess Through Pure Neural Intuition

A neural network that plays chess without search algorithms—relying purely on learned intuition from millions of positions evaluated by Stockfish.

**Inspired by:** Google DeepMind's [*"Grandmaster-Level Chess Without Search"*](https://arxiv.org/html/2402.04494v1)

---

## 🎯 About

This project demonstrates that neural networks can develop chess intuition without performing any search. The model:

- **Evaluates positions** based on centipawn (cp) scores from deep Stockfish analysis
- **Predicts optimal moves** directly from FEN notation
- **Learns strategic patterns** from millions of positions

No minimax. No alpha-beta pruning. Pure neural intuition.

---

## 📊 Dataset

### Original training data from **Lichess** via HuggingFace:
- [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
- ~784M of positions with Stockfish deep analysis

### Transformer, ready-to-Use Training Dataset (HuggingFace)

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

**Note:** This dataset corresponds to the output of the data processing pipeline's final stage (`scripts/stage_2_deduplicate_data.py`), which performs global deduplication across all files. The 10-part split was created for convenient distribution and incremental loading.

### Data Processing Pipeline

The raw Lichess data undergoes a two-stage deduplication process:

1. **Stage 1** (`scripts/stage_1_deduplicate_data.py`): File-level deduplication
   - Input: ~784M positions with duplicates
   - Output: ~400M positions (unique per file)

2. **Stage 2** (`scripts/stage_2_deduplicate_data.py`): Global deduplication
   - Input: Stage 1 output
   - Output: **316M globally unique positions** → Split into 10 parts and published to HuggingFace

For full pipeline details, see the `scripts/` directory.

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

**Principal AI/ML Engineer**

LinkedIn: [Mateusz Grzyb](https://www.linkedin.com/in/mateusz--grzyb/)
