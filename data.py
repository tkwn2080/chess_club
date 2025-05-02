#!/usr/bin/env python3
"""
Chess Neural Network Competition - Data Generation
This script downloads and processes chess game data from Lichess,
filtering for high-rated games (2200+ Elo) to create training data.
"""

import os
import io
import sys
import gzip
import time
import chess
import chess.pgn
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import datetime

# Configuration
MIN_ELO = 2200  # Minimum Elo rating for games
DATA_DIR = "data"
LICHESS_DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_current_month.pgn.zst"
MAX_GAMES = 10000  # Maximum number of games to process
OUTPUT_PGN = os.path.join(DATA_DIR, "games.pgn")
OUTPUT_CSV = os.path.join(DATA_DIR, "positions.csv")
OUTPUT_NPZ = os.path.join(DATA_DIR, "train.npz")

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

def download_chess_database():
    """Download a lightweight version of Lichess database."""
    print(f"Downloading chess database from {LICHESS_DB_URL}")
    
    # For demonstration, we'll use a smaller dataset
    # In reality, you might want to use a more specific URL or approach
    response = requests.get("https://lichess.org/api/games/user/DrNykterstein", 
                           params={"max": 100, "perfType": "rapid", "pgnInJson": "false"},
                           headers={"Accept": "application/x-chess-pgn"})
    
    if response.status_code == 200:
        with open(OUTPUT_PGN, 'w') as f:
            f.write(response.text)
        print(f"Downloaded sample data to {OUTPUT_PGN}")
        return True
    else:
        print(f"Failed to download: {response.status_code}")
        return False

def filter_high_elo_games():
    """Filter games for those with both players rated above MIN_ELO."""
    print(f"Filtering games with players rated {MIN_ELO}+ Elo")
    
    high_elo_games = []
    games_processed = 0
    
    with open(OUTPUT_PGN, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
                
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            
            if white_elo >= MIN_ELO and black_elo >= MIN_ELO:
                high_elo_games.append(game)
            
            games_processed += 1
            if games_processed % 100 == 0:
                print(f"Processed {games_processed} games, kept {len(high_elo_games)}")
            
            if len(high_elo_games) >= MAX_GAMES:
                break
    
    print(f"Found {len(high_elo_games)} games with both players rated {MIN_ELO}+")
    return high_elo_games

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
        # Skip opening moves as they're less informative
        sample_indices = list(range(min(8, len(moves)), len(moves), 2))
        
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
    """Main function to download and process chess data."""
    try:
        print("Chess Neural Network Competition - Data Generation")
        start_time = time.time()
        
        # Create data directory
        ensure_data_dir()
        
        # Download data
        if not os.path.exists(OUTPUT_PGN) or os.path.getsize(OUTPUT_PGN) == 0:
            if not download_chess_database():
                print("Failed to download chess database. Exiting.")
                return
        else:
            print(f"Using existing data file: {OUTPUT_PGN}")
        
        # Filter games
        high_elo_games = filter_high_elo_games()
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
