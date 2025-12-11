# 🚀 Model Loading Guide

This guide covers three main use cases for loading and using the pre-trained models.

---

## 1️⃣ Loading ResNet/CNN Models (Standard Keras)

ResNet and CNN models use standard Keras layers and can be loaded directly without any custom objects.

### **For Prediction (Inference)**

```python
import keras

from src.data_preparation.data_processing import fen_to_tensor

# Load the model
model = keras.models.load_model('03_resnet_l/model.keras')

# Prepare your input (8x8x12 board representation)
# Example: position tensor from FEN
position = fen_to_tensor(fen_string)  # Shape: (8, 8, 12)
position_batch = np.expand_dims(position, axis=0)   # Shape: (1, 8, 8, 12)

evaluation = model.predict(position_batch)
print(f"Position evaluation: {evaluation[0][0]}")
# Note: Unlike Stockfish's centipawn scores, this model outputs normalized values:
# -1.0 to 1.0 range, where:
#   < 0: Position is losing for the side to move
#   ≈ 0: Position is roughly equal
#   > 0: Position is winning for the side to move
```

### **For Fine-tuning (Training)**

```python
import keras

# Load model with optimizer state preserved
model = keras.models.load_model('03_resnet_l/model.keras')

# Model is ready for continued training
# Optimizer state and learning rate schedule are preserved

# Continue training on your data
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[...]
)
```

### **Load Weights Only (Using Repository Architecture)**

If you want to rebuild the model from source code instead of loading the full .keras file:

```python
import keras
from src.models.resnet import build_resnet  # this function is in the repo; check example of training process ==> scripts/training/model_03_resnet_l.py

# Build fresh model from scratch
model = build_resnet(
    input_shape=(8, 8, 12),
    filters=256,
    dense_shape=512,
    n_res_blocks=10,
    l2_value=1e-4,
    use_batchnorm=True,
    use_squeeze_and_excitation=False,
    dropout_final=0.5
)

# Load pre-trained weights
model.load_weights('03_resnet_l/weights.h5')

# Compile with your preferred optimizer/loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)
```

## 2️⃣ Loading Vision Transformer Models (ViT)

ViT models require a custom learning rate scheduler (`WarmUpCosineDecay`) that was used during training. You must provide this class when loading.

### **For Prediction (Inference)**

```python
import keras
import numpy as np

from src.utils.tools import WarmUpCosineDecay
from src.data_preparation.data_processing import fen_to_tensor

# Load model with custom learning rate scheduler
model = keras.models.load_model(
    '06_vit_m/model.keras',
    custom_objects={'WarmUpCosineDecay': WarmUpCosineDecay}
)

# Prepare your input (8x8x12 board representation)
position = fen_to_tensor(fen_string)  # Shape: (8, 8, 12)
position_batch = np.expand_dims(position, axis=0)   # Shape: (1, 8, 8, 12)

# Get position evaluation
evaluation = model.predict(position_batch)
print(f"Position evaluation: {evaluation[0][0]}")
# Note: Model outputs normalized scores (-1 to 1), not centipawns
# See "Understanding Model Output" section below
```

### **For Fine-tuning (Training)**

```python
import keras
from src.utils.tools import WarmUpCosineDecay

# Load model with custom objects
model = keras.models.load_model(
    '06_vit_m/model.keras',
    custom_objects={'WarmUpCosineDecay': WarmUpCosineDecay}
)

# The WarmUpCosineDecay scheduler state is preserved
# You can continue training with the same schedule or modify it

# Option 1: Continue with existing scheduler
model.fit(train_dataset, epochs=10)

# Option 2: Replace with new optimizer/scheduler
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.0001),
    loss='mse'
)
model.fit(train_dataset, epochs=10)
```

### **Load Weights Only (Custom Architecture)**

```python
import keras
from src.models.vision_transformer import build_vit
from src.utils.tools import WarmUpCosineDecay

# Build fresh ViT architecture
model = build_vit(
    input_shape=(8, 8, 12),
    projection_dim=256,
    num_heads=4,
    transformer_layers=5,
    dropout_rate=0.1
)

# Load pre-trained weights (no custom_objects needed for weights only)
model.load_weights('06_vit_m/weights.h5')

lr_schedule = WarmUpCosineDecay(
    initial_learning_rate=0.0001,
    warmup_steps=1000,
    total_steps=100000
)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule),
    loss='mse'
)
```

**Note:** The `WarmUpCosineDecay` class is located in `src/utils/tools.py` in the main repository. You'll need to clone the repo or copy this class to use ViT models.

---

## 3️⃣ Playing Chess with ChessAI Wrapper

The `ChessAI` class provides a high-level interface for playing chess with the models using the `python-chess` library.
PS. Compleate example of game is described in repo: `notebooks/01_testing_chessai_class.ipynb`

### **Basic Game Example**

```python
import chess
from src.chess_ai.core.model import ChessAI

# Initialize the chess engine with any model
chess_bot = ChessAI('06_vit_m/model.keras')

# Create a new game
board = chess.Board()

# Play a few moves
# Move #1
player_move = board.parse_san("e4")  # Player plays e2-e4
board.push(player_move)
engine_move = chess_bot.make_move(board)  # Engine responds
board.push(engine_move)
print(board)

# Move #2
player_move = board.parse_san("d4")  # Player plays d2-d4
board.push(player_move)
engine_move = chess_bot.make_move(board)  # Engine responds
board.push(engine_move)
print(board)
```

### **Full Game Loop**

```python
import chess
from src.chess_ai.core.model import ChessAI

# Initialize engine
chess_bot = ChessAI('06_vit_m/model.keras')
board = chess.Board()

# Play until game over
while not board.is_game_over():
    if board.turn == chess.WHITE:
        # Human plays white
        move_input = input("Your move (e.g., e2e4): ")
        try:
            move = chess.Move.from_uci(move_input)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Illegal move! Try again.")
                continue
        except ValueError:
            print("Invalid move format! Use UCI notation (e.g., e2e4)")
            continue
    else:
        # Engine plays black
        engine_move = chess_bot.make_move(board)
        board.push(engine_move)
        print(f"Engine plays: {engine_move.uci()}")
    
    print("\n" + str(board) + "\n")

# Game over
print("Game Over!")
print(f"Result: {board.result()}")
```

### **Jupyter Notebook Display**

If you're using Jupyter notebooks, you can display the board visually:

```python
import chess
import chess.svg
from IPython.display import display, SVG
from src.chess_ai.core.model import ChessAI

chess_bot = ChessAI('06_vit_m/model.keras')
board = chess.Board()

# Move #1
player_move = board.parse_san("e4")
board.push(player_move)
engine_move = chess_bot.make_move(board)
board.push(engine_move)

# Display board as SVG
display(SVG(chess.svg.board(board=board, size=400)))
```

### **Advanced: Position Analysis**

```python
from src.chess_ai.core.model import ChessAI
import chess

chess_bot = ChessAI('06_vit_m/model.keras')
board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")

# Get the best move
best_move = chess_bot.make_move(board)
print(f"Best move: {best_move.uci()}")

# You can also access the underlying model for raw evaluation
position_tensor = chess_bot.board_to_tensor(board)
evaluation = chess_bot.model.predict(position_tensor)
print(f"Position evaluation: {evaluation[0][0]:.3f} (range: -1 to 1)")
print(f"Interpretation: {'Winning' if evaluation[0][0] > 0 else 'Losing'} for {board.turn}")
```

### **Comparing Different Models**

```python
import chess
from src.chess_ai.core.model import ChessAI

# Load multiple models
vit_bot = ChessAI('06_vit_m/model.keras')
resnet_bot = ChessAI('03_resnet_l/model.keras')

board = chess.Board()

# See how different models evaluate the same position
board.push_san("e4")
board.push_san("e5")

vit_move = vit_bot.make_move(board)
resnet_move = resnet_bot.make_move(board)

print(f"ViT-Medium suggests: {vit_move.uci()}")
print(f"ResNet-Large suggests: {resnet_move.uci()}")
```

## 📐 Understanding Model Output

### **Evaluation Format**

**Important:** These models output **normalized evaluations** in the range `[-1, 1]`, NOT centipawn scores like Stockfish.

```python
evaluation = model.predict(position_batch)[0][0]

# Interpretation (always from side-to-move perspective):
if evaluation < -0.5:
    print("Position is significantly losing")
elif evaluation < 0:
    print("Position is slightly losing")
elif evaluation < 0.5:
    print("Position is slightly winning")
else:
    print("Position is significantly winning")
```

### **Key Differences from Stockfish**

| Aspect | Stockfish | This Model |
|--------|-----------|------------|
| **Output range** | -∞ to +∞ (centipawns) | -1 to +1 (normalized) |
| **Perspective** | Always from White's view | From side-to-move view |
| **Equal position** | 0 cp | ≈ 0.0 |
| **Winning** | +300 cp | ≈ +0.3 to +0.7 |
| **Mate advantage** | +10000 cp | ≈ +0.9 to +1.0 |

### **Why Normalized Output?**

The model was trained on Stockfish evaluations that were **normalized** during preprocessing:

```python
# Training preprocessing (for reference)
normalized_cp = tanh(stockfish_cp / 1000)  # Maps ±∞ to [-1, 1]
```

This provides:
- ✅ Bounded output range (stable gradients)
- ✅ Better convergence during training
- ✅ Easier interpretation for humans

### **Converting to Approximate Centipawns**

If you need rough centipawn equivalents:

```python
import numpy as np

def model_to_centipawns(normalized_eval):
    """
    Approximate conversion from normalized [-1, 1] to centipawns.
    Note: This is a rough estimate, not exact Stockfish equivalence.
    """
    return np.arctanh(np.clip(normalized_eval, -0.99, 0.99)) * 1000

# Example
normalized = 0.35
approx_cp = model_to_centipawns(normalized)
print(f"Normalized: {normalized:.2f} ≈ {approx_cp:.0f} centipawns")
# Output: Normalized: 0.35 ≈ 365 centipawns
```

⚠️ **Warning:** This conversion is approximate. The model's output doesn't perfectly align with Stockfish centipawns, especially in extreme positions.

## 📝 Notes

### **ViT Models and Custom Objects**
- ViT models use `WarmUpCosineDecay` learning rate scheduler
- Always include `custom_objects={'WarmUpCosineDecay': WarmUpCosineDecay}` when loading full `.keras` models
- Not needed when loading only weights (`.h5` files)

### **ChessAI Class Requirements**
- Requires `python-chess` library for board management
- Automatically handles FEN ↔ tensor conversion
- Works with any model (ResNet, ViT, CNN)
- Returns `chess.Move` objects compatible with `python-chess`

### **Model Input Format**
All models expect:
- **Input shape:** `(batch_size, 8, 8, 12)`
- **8x8:** Chess board dimensions
- **12 channels:** 6 piece types × 2 colors (one-hot encoded)

### **Model Output Format**
- **Output shape:** `(batch_size, 1)`
- **Value range:** `-1.0` to `+1.0` (normalized evaluation)
- **Perspective:** Always from side-to-move viewpoint
  - `evaluation < 0`: Side to move is losing
  - `evaluation ≈ 0`: Position is roughly equal
  - `evaluation > 0`: Side to move is winning

**Not Stockfish centipawns!** See "Understanding Model Output" section above for details.

## 🐛 Troubleshooting

### **"Unknown layer: WarmUpCosineDecay"**
```python
# Solution: Import and provide custom_objects when loading ViT
from src.utils.tools import WarmUpCosineDecay
model = keras.models.load_model('06_vit_m/model.keras', 
                                custom_objects={'WarmUpCosineDecay': WarmUpCosineDecay})
```

### **"Module 'src' not found"**
```bash
# Solution: Clone the repository and install dependencies
git clone https://github.com/mateuszgrzyb-pl/searchless-chess.git
cd searchless-chess
pip install -r requirements.txt
```

### **ChessAI import error**
```python
# Make sure you're in the project root directory
import sys
sys.path.append('/path/to/searchless-chess')
from src.chess_ai.core.model import ChessAI
```

## 📚 Additional Resources

- **Full documentation:** [Main README](https://github.com/mateuszgrzyb-pl/searchless-chess)
- **Evaluation code:** See `notebooks/03_testing_model_elo_on_lichess_puzzles.ipynb`
- **Training scripts:** See `scripts/training/` to understand trainig process
- **Model's architecture:** See `src/models` for architecture definitions
