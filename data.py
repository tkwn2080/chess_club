#!/usr/bin/env python3
"""
Chess Neural Network Competition - Data Generation

This script downloads and processes chess game data from Lichess,
filtering for high-rated games (2200+ Elo) to create training data.

TRAINING DATA FORMAT:
The output file (data/train.npz) contains:
- X: Board state features (numpy array of shape [n_positions, 768])
  - Each position is encoded as a 768-dimensional vector (64 squares × 12 piece types)
  - Piece types: [white pawn, knight, bishop, rook, queen, king, black pawn, knight, bishop, rook, queen, king]
  - Value is 1 if piece is present, 0 otherwise

- y_move: Next move in UCI format (list of strings)
  - Example: "e2e4", "g1f3", etc.
  - These need to be encoded to numerical format for neural network training

- y_value: Game outcome from current player's perspective (numpy array)
  - +1: Current player wins
  - 0: Draw
  - -1: Current player loses

The data includes positions from all game phases (opening, middlegame, endgame)
from games where both players are rated 2200+ Elo.
"""

import os
import io
import sys
import time
import chess
import chess.pgn
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import zstandard as zstd

# Configuration
MIN_ELO = 1800  # Minimum Elo rating for games
DATA_DIR = "data"
LICHESS_DB_FILE = "lichess_db_standard_rated_2014-09.pgn.zst"
LICHESS_DB_URL = f"https://database.lichess.org/standard/{LICHESS_DB_FILE}"
MAX_GAMES = 100000  # Maximum number of games to process
INPUT_PGN_ZST = os.path.join(DATA_DIR, LICHESS_DB_FILE)
OUTPUT_PGN = os.path.join(DATA_DIR, "games.pgn")
OUTPUT_CSV = os.path.join(DATA_DIR, "positions.csv")
OUTPUT_NPZ = os.path.join(DATA_DIR, "train.npz")

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def download_lichess_database():
    """Download the Lichess database if it doesn't exist."""
    if os.path.exists(INPUT_PGN_ZST):
        print(f"Database file already exists: {INPUT_PGN_ZST}")
        return True

    print(f"Downloading {LICHESS_DB_FILE} from Lichess...")
    print("This is a large file (~600MB) and may take several minutes.")

    try:
        response = requests.get(LICHESS_DB_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(INPUT_PGN_ZST, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=LICHESS_DB_FILE) as pbar:
                for chunk in response.iter_content(block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"Successfully downloaded {LICHESS_DB_FILE}")
        return True

    except Exception as e:
        print(f"Error downloading database: {e}")
        if os.path.exists(INPUT_PGN_ZST):
            os.remove(INPUT_PGN_ZST)
        return False

def extract_from_compressed_pgn():
    """Extract high-rated games from compressed PGN file."""
    print(f"Processing {LICHESS_DB_FILE}...")

    if not os.path.exists(INPUT_PGN_ZST):
        print(f"Error: File {INPUT_PGN_ZST} not found!")
        print(f"Please download it from https://database.lichess.org/")
        return []

    high_elo_games = []
    games_processed = 0

    try:
        # Open the compressed file
        with open(INPUT_PGN_ZST, 'rb') as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break

                    games_processed += 1

                    # Get Elo ratings
                    try:
                        white_elo_str = game.headers.get("WhiteElo", "0")
                        black_elo_str = game.headers.get("BlackElo", "0")

                        # Remove any non-numeric characters (like trailing ?)
                        white_elo = int(''.join(c for c in white_elo_str if c.isdigit()) or "0")
                        black_elo = int(''.join(c for c in black_elo_str if c.isdigit()) or "0")
                    except (ValueError, TypeError):
                        white_elo = 0
                        black_elo = 0

                    if white_elo >= MIN_ELO and black_elo >= MIN_ELO:
                        high_elo_games.append(game)

                    if games_processed == 1:
                        print(f"First game headers: {game.headers}")
                        print(f"White Elo: {white_elo}, Black Elo: {black_elo}")

                    if games_processed % 1000 == 0:
                        print(f"Processed {games_processed} games, found {len(high_elo_games)} high-rated")

                    if len(high_elo_games) >= MAX_GAMES:
                        print(f"Reached maximum game limit ({MAX_GAMES})")
                        break

        print(f"Processed {games_processed} games total, found {len(high_elo_games)} with both players rated {MIN_ELO}+")
        return high_elo_games

    except Exception as e:
        print(f"Error processing compressed file: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_filtered_games(games):
    """Save filtered games to a new PGN file."""
    with open(OUTPUT_PGN, 'w') as out_file:
        for game in games:
            out_file.write(str(game) + "\n\n")
    print(f"Saved filtered games to {OUTPUT_PGN}")

def extract_positions(games):
    """Extract board positions and results from games."""
    positions = []
    results = {'1-0': 1, '0-1': -1, '1/2-1/2': 0, '*': None}

    for game in tqdm(games, desc="Extracting positions"):
        result = results[game.headers.get("Result", "*")]
        if result is None:  # Skip unfinished games
            continue

        board = game.board()
        moves = list(game.mainline_moves())

        # Sample positions from the game (not every position to reduce dataset size)
        # Include opening moves to help neural network learn opening principles
        # Sample every other move to reduce dataset size while maintaining game flow
        sample_indices = list(range(0, len(moves), 2))

        for i in sample_indices:
            if i >= len(moves):
                break

            # Apply moves up to this position
            board = game.board()
            for j in range(i):
                board.push(moves[j])

            # Get board state and legal moves
            fen = board.fen()
            next_move = moves[i].uci() if i < len(moves) else None

            # Add to dataset
            positions.append({
                'fen': fen,
                'next_move': next_move,
                'result': result,
                'turn': 'w' if board.turn == chess.WHITE else 'b'
            })

    return positions

def convert_to_training_format(positions_df):
    """Convert positions dataframe to numpy arrays for training."""
    # This is a simplified example - you would implement your own board encoding
    # based on your neural network architecture

    # For example, a simple one-hot encoding for piece positions
    X = []  # Board states
    y_move = []  # Move probabilities
    y_value = []  # Game outcome prediction

    print("Converting positions to training format...")

    # Simple placeholder implementation - this would be more complex in practice
    for _, row in tqdm(positions_df.iterrows(), total=len(positions_df)):
        # Convert FEN to a simple feature vector (placeholder implementation)
        board = chess.Board(row['fen'])
        features = np.zeros(64 * 12)  # 64 squares, 12 piece types (6 pieces * 2 colors)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                features[square * 12 + piece_idx] = 1

        X.append(features)

        # Move encoding (simplified - just storing UCI string)
        y_move.append(row['next_move'])

        # Value target (game result from perspective of current player)
        value = row['result'] if row['turn'] == 'w' else -row['result']
        y_value.append(value)

    # Convert to numpy arrays
    X = np.array(X)
    y_value = np.array(y_value)

    # Save to npz file
    np.savez(OUTPUT_NPZ, X=X, y_move=y_move, y_value=y_value)
    print(f"Saved training data to {OUTPUT_NPZ}")

def main():
    """Main function to extract and process chess data."""
    try:
        print("Chess Neural Network Competition - Data Generation")
        start_time = time.time()

        # Create data directory
        ensure_data_dir()

        # Download database if needed
        if not download_lichess_database():
            print("Failed to download database. Exiting.")
            return 1

        # Extract games from compressed file
        high_elo_games = extract_from_compressed_pgn()

        if not high_elo_games:
            print("No high-rated games found. Exiting.")
            return 1

        # Save filtered games to PGN
        save_filtered_games(high_elo_games)

        # Extract positions
        positions = extract_positions(high_elo_games)
        positions_df = pd.DataFrame(positions)
        positions_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(positions_df)} positions to {OUTPUT_CSV}")

        # Convert to training format
        convert_to_training_format(positions_df)

        elapsed_time = time.time() - start_time
        print(f"Data generation completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

# Note: The current board encoding is simple and can be improved.
# More sophisticated encodings might include:
# - Bitboards for each piece type
# - En passant squares
# - Castling rights
# - Move count / 50-move rule counter
# - Side to move
# - Previous board positions (for detecting repetitions)
# - Legal move mask
#
# The move encoding (currently just UCI strings) should be converted to:
# - One-hot encoding over all possible moves (4096 dimensions)
# - Or piece-centric encoding (from_square + to_square + promotion)
#
# For serious neural network training, consider using existing
# chess ML libraries like python-chess-nn or leela-chess encodings.
