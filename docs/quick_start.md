## 🚀 Quick Start

### Download Pre-trained Models

All models are available via [GitHub Releases](https://github.com/mateuszgrzyb-pl/searchless-chess/releases/latest):

```bash
# Download the best model (ViT-Medium)
wget https://github.com/mateuszgrzyb-pl/searchless-chess/releases/download/v1.0.0/05-vit-small-v1.0.0.zip
unzip vit-medium-v1.0.0.zip
```

Or download directly from the [Releases page](https://github.com/mateuszgrzyb-pl/searchless-chess/releases).

### Basic Usage

```python
import chess
from src.chess_ai.core.model import ChessAI

# Load the chess engine (works with any model)
chess_bot = ChessAI('vit_medium/model.keras')

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
