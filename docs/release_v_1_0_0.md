# 🎉 Searchless Chess v1.0.0 - Initial Model Release

First public release of all trained models from the Searchless Chess project!

## 🏆 Highlights

- **6 pre-trained models** achieving up to ~1960 ELO without search
- **Vision Transformers** demonstrating 10x parameter efficiency over ResNets
- **316M training positions** from Lichess + Stockfish evaluations
- **200+ hours** of training on NVIDIA A100

## 📦 What's Included

| Model | Size | ELO | Download |
|-------|------|-----|----------|
| **ViT-Medium** ⭐ | ~199MB | 1960 | [vit-medium-v1.0.0.zip] |
| **ViT-Small** | ~56MB | 1817 | [vit-small-v1.0.0.zip] |
| ResNet-XL | ~431MB | 1719 | [resnet-xl-v1.0.0.zip] |
| ResNet-L | ~135MB | 1711 | [resnet-l-v1.0.0.zip] |
| ResNet-M | ~42MB | 1515 | [resnet-m-v1.0.0.zip] |
| CNN Baseline | ~14MB | 1112 | [cnn-s-v1.0.0.zip] |

Note, the packages are quite large because each of them contains four files: the model in Keras format, the H5 model, the raw weights, a configuration file, and a file with the model’s metrics. You can load model from: "model.keras" or "model.weights.h5".

## 🚀 Quick Start

### Download Pre-trained Models

All models are available via [GitHub Releases](https://github.com/mateuszgrzyb-pl/searchless-chess/releases/latest):

```bash
# Download the best model (ViT-Medium)
wget https://github.com/mateuszgrzyb-pl/searchless-chess/releases/download/v1.0.0/vit-medium-v1.0.0.zip
unzip vit-medium-v1.0.0.zip
```

Or download directly from the [Releases page](https://github.com/mateuszgrzyb-pl/searchless-chess/releases).

### Basic Usage

```python
import chess
from src.chess_ai.core.model import ChessAI

# Load the chess engine (works with any model)
chess_bot = ChessAI('06_vit_m/model.keras')

# Create a chess board
board = chess.Board()

# Make moves
board.push_san("e4")  # Your move
engine_move = chess_bot.make_move(board)  # Engine responds
board.push(engine_move)

print(board)
print(f"Engine played: {engine_move}")
```

**Note:** Model evaluations are in the range `[-1, 1]` (normalized scores), not centipawns. Values represent position quality from the side-to-move perspective.

### Advanced Loading Options

For detailed examples including:
- Loading ResNet/CNN models (standard Keras)
- Loading ViT models (requires custom objects)
- Direct model inference without ChessAI wrapper
- Fine-tuning and transfer learning

See the complete guide: **[docs/model_loading_guide.md](docs/model_loading_guide.md)**

---

## 📊 Performance Summary

- **ViT-Medium:** 1960 ELO, 88% accuracy on intermediate puzzles
- **ViT-Small:** 1817 ELO, 82% accuracy, 2.64M parameters
- **Parameter Efficiency:** ViT-Small = 688 ELO/1M params vs ResNet-XL = 70

## 🔬 What's New

- Full model architectures (.keras format)
- Standalone weights (.h5 format)
- Training configurations (config.json)
- Performance metrics (metrics.json)
- Complete evaluation data from 1,200 Lichess puzzles

## 📖 Documentation

Full project documentation: [README.md](https://github.com/mateuszgrzyb-pl/searchless-chess)

**Dataset:** [HuggingFace](https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized)

## 🐛 Known Issues

None reported yet! Please open an issue if you encounter problems.

## 📝 Citation
```bibtex
@software{grzyb2025searchless,
  author = {Grzyb, Mateusz},
  title = {Searchless Chess: Master-Level Chess Through Pure Neural Intuition},
  year = {2025},
  url = {https://github.com/mateuszgrzyb-pl/searchless-chess}
}
```

## 🙏 Acknowledgments

- Google DeepMind for inspiration ([Searchless Chess paper](https://arxiv.org/html/2402.04494v1))
- Lichess for the evaluation dataset
- The open-source ML community

## 📧 Contact

Questions? Reach out on [LinkedIn](https://www.linkedin.com/in/mateusz--grzyb/)

---

**⭐ If you find this useful, please star the repo!**
