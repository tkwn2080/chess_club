# Chess Club for Neural Networks

Both parties are given the same database, they have one hour to build and train a neural network to play chess.

## Overview

This repository contains everything needed to:
1. Generate a training dataset from high-level chess games (2200+ Elo)
2. Provide a simple UI for pitting neural networks against each other or human players
3. Connect your own chess neural network using any ML framework

![ap8510100244-1726075f58d7fd928d17f0df51b61a97a2a70afc](https://github.com/user-attachments/assets/3b3f8d0e-dedd-4643-b52a-4923db9fa0ca)

## Getting Started

### Prerequisites
- Python 3.8+
- pip (for installing dependencies)

```bash
# Install dependencies
pip install -r requirements.txt
```

### Generating Training Data

Run the data generation script to download and process a chess database:

```bash
python data.py
```

This will:
1. Automatically download the Lichess database from September 2014 (~600MB)
2. Extract and process games where both players are rated 2200+ Elo
3. Create training data files ready for neural network training
4. Process up to 100,000 high-rated games

### Playing Games

To start the chess interface:

```bash
python game.py
```

1. The UI will appear with a chess board
2. Click "W" or "B" to select which side your neural network will play
3. Make moves by clicking on a piece and then its destination square

## Connecting Your Neural Network

Your neural network should implement this simple interface:

```python
class ChessModel:
    def __init__(self, model_path=None):
        # Initialize your model here
        pass
        
    def predict_move(self, board_state):
        """
        Args:
            board_state: A FEN string representing the current board state
            
        Returns:
            move: A move in UCI format (e.g., "e2e4")
        """
        # Your model inference code here
        pass
```

Example implementation for different frameworks:

### PyTorch Example
```python
import torch
import chess

class PyTorchChessModel(ChessModel):
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load your model
        self.model = YourModelClass().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict_move(self, board_state):
        # Convert FEN to your model's input format
        board = chess.Board(board_state)
        # Your preprocessing code
        # ...
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
        # Convert output to move
        # ...
        return move
```

### TensorFlow Example
```python
import tensorflow as tf
import chess

class TensorFlowChessModel(ChessModel):
    def __init__(self, model_path=None):
        # Load your model
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = YourModelClass()
    
    def predict_move(self, board_state):
        # Convert FEN to your model's input format
        board = chess.Board(board_state)
        # Your preprocessing code
        # ...
        # Get model prediction
        output = self.model.predict(input_array)
        # Convert output to move
        # ...
        return move
```

## Competition Setup

For a competition between two neural networks:

1. Set up two computers side by side with a physical chess board
2. On computer A, run `python game.py` and click "W"
3. On computer B, run `python game.py` and click "B"
4. Each computer will display the board state and make moves for its assigned color
5. Players transfer the moves between the physical board and computers

## Data Format

The data.py script generates three files in the `data/` directory:

### 1. `games.pgn`
Raw PGN files from filtered games (both players 2200+ Elo)

### 2. `positions.csv`
Board positions with the following columns:
- `fen`: Board state in FEN notation
- `next_move`: Next move in UCI format (e.g., "e2e4")
- `result`: Game outcome (1=white wins, -1=black wins, 0=draw)
- `turn`: Side to move ('w' or 'b')

### 3. `train.npz`
NumPy arrays ready for neural network training:
- `X`: Board states as 768-dimensional vectors (shape: [n_positions, 768])
  - Each position: 64 squares × 12 piece types (6 pieces × 2 colors)
  - Binary encoding: 1 if piece present, 0 otherwise
- `y_move`: Next moves in UCI format (list of strings)
  - Needs conversion to numerical format for NN training
- `y_value`: Game outcomes from current player's perspective (numpy array)
  - +1: Current player wins
  - 0: Draw
  - -1: Current player loses

Note: The data includes positions from all game phases (opening, middlegame, endgame), sampling every other move to reduce dataset size while maintaining game flow.

## Customization

You can modify the Elo threshold by editing data.py:

```python
# Change 2200 to your desired Elo threshold
MIN_ELO = 2200
```

## License

[MIT License](LICENSE)
