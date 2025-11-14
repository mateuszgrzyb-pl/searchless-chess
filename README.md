# ♟️ Searchless Chess: Grandmaster-Level Chess Through Pure Neural Intuition

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

Training data from **Lichess** via HuggingFace:
- [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
- Millions of positions with Stockfish deep analysis

---

## 🚀 Installation

```bash
git clone https://github.com/mateuszgrzyb-pl/searchless-chess.git
cd searchless-chess
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