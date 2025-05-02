#!/usr/bin/env python3
"""
Chess Neural Network Competition - Game Interface
This script provides a simple GUI for playing chess against a neural network.
"""

import os
import sys
import time
import importlib.util
import chess
import pygame
import pygame.freetype
from pygame.locals import *
import importlib.util
import numpy as np

# Configuration
WINDOW_SIZE = (800, 600)
BOARD_SIZE = 512  # Must be divisible by 8
SQUARE_SIZE = BOARD_SIZE // 8
MOVE_HIGHLIGHT_COLOR = (124, 252, 0)  # Light green
SELECTED_HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow
LAST_MOVE_HIGHLIGHT_COLOR = (135, 206, 250)  # Light blue
TEXT_COLOR = (255, 255, 255)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Pygame initialization
pygame.init()
pygame.freetype.init()
font = pygame.freetype.SysFont('Arial', 24)
small_font = pygame.freetype.SysFont('Arial', 16)
piece_images = {}

# Model interface
class DummyModel:
    """Placeholder model that makes random legal moves."""
    def __init__(self):
        self.name = "Random Mover"
        
    def predict_move(self, board_state):
        """Return a random legal move given a board state in FEN."""
        board = chess.Board(board_state)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return legal_moves[np.random.randint(0, len(legal_moves))].uci()

class ChessGame:
    """Main game class managing the chess board and UI."""
    
    def __init__(self):
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Chess Neural Network Competition")
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.selected_square = None
        self.highlighted_squares = []
        self.last_move = None
        self.player_color = None  # None, chess.WHITE, or chess.BLACK
        self.model = None
        self.model_status = "No model loaded"
        self.message = "Select which side your neural network will play (W/B)"
        self.load_piece_images()
        
    def load_piece_images(self):
        """Load chess piece images."""
        pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        for piece in pieces:
            # Using a simple colored rectangle as a placeholder
            # In a real implementation, you would load actual piece images
            image = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            color = (200, 200, 200) if piece.isupper() else (100, 100, 100)
            pygame.draw.rect(image, color, (10, 10, SQUARE_SIZE-20, SQUARE_SIZE-20))
            
            # Add a letter in the middle to identify the piece
            text_surf, _ = font.render(piece, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(SQUARE_SIZE//2, SQUARE_SIZE//2))
            image.blit(text_surf, text_rect)
            
            piece_images[piece] = image
    
    def load_model(self, model_path=None):
        """
        Load a chess neural network model.
        If no model is provided, use the dummy model.
        """
        try:
            # Check if a custom model is available
            spec = importlib.util.find_spec("custom_model")
            if spec is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'ChessModel'):
                    self.model = module.ChessModel(model_path)
                    self.model_status = "Custom model loaded"
                    return
            
            # Fallback to dummy model
            self.model = DummyModel()
            self.model_status = "Using dummy model (random moves)"
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = DummyModel()
            self.model_status = f"Error: {str(e)[:50]}... Using dummy model"
    
    def draw_board(self):
        """Draw the chess board."""
        # Draw board squares
        for row in range(8):
            for col in range(8):
                x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
                square = chess.square(col, 7 - row)  # Convert to chess.Square
                color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
                
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # Highlight selected square
                if self.selected_square == square:
                    pygame.draw.rect(self.screen, SELECTED_HIGHLIGHT_COLOR, 
                                    (x, y, SQUARE_SIZE, SQUARE_SIZE), 3)
                
                # Highlight legal moves
                if square in self.highlighted_squares:
                    pygame.draw.rect(self.screen, MOVE_HIGHLIGHT_COLOR, 
                                    (x, y, SQUARE_SIZE, SQUARE_SIZE), 3)
                
                # Highlight last move
                if self.last_move and (square == self.last_move.from_square or 
                                      square == self.last_move.to_square):
                    pygame.draw.rect(self.screen, LAST_MOVE_HIGHLIGHT_COLOR, 
                                    (x, y, SQUARE_SIZE, SQUARE_SIZE), 3)
        
        # Draw coordinate labels
        for i in range(8):
            # Rank numbers (1-8)
            text_surf, _ = small_font.render(str(8 - i), TEXT_COLOR)
            self.screen.blit(text_surf, (BOARD_SIZE + 10, i * SQUARE_SIZE + SQUARE_SIZE//2 - 8))
            
            # File letters (a-h)
            text_surf, _ = small_font.render(chr(97 + i), TEXT_COLOR)
            self.screen.blit(text_surf, (i * SQUARE_SIZE + SQUARE_SIZE//2 - 5, BOARD_SIZE + 10))
    
    def draw_pieces(self):
        """Draw chess pieces based on the current board state."""
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)  # Convert to chess.Square
                piece = self.board.piece_at(square)
                if piece:
                    piece_symbol = piece.symbol()
                    x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
                    self.screen.blit(piece_images[piece_symbol], (x, y))
    
    def draw_status(self):
        """Draw game status information."""
        # Background for text area
        pygame.draw.rect(self.screen, (50, 50, 50), 
                         (BOARD_SIZE, 0, WINDOW_SIZE[0] - BOARD_SIZE, BOARD_SIZE))
        
        # Game status
        turn_text = "White's turn" if self.board.turn == chess.WHITE else "Black's turn"
        status_surf, status_rect = font.render(turn_text, TEXT_COLOR)
        self.screen.blit(status_surf, (BOARD_SIZE + 20, 20))
        
        # Check/checkmate status
        if self.board.is_checkmate():
            result = "0-1" if self.board.turn == chess.WHITE else "1-0"
            check_text = f"Checkmate! {result}"
        elif self.board.is_check():
            check_text = "Check!"
        elif self.board.is_stalemate():
            check_text = "Stalemate! Draw."
        elif self.board.is_insufficient_material():
            check_text = "Insufficient material! Draw."
        elif self.board.can_claim_draw():
            check_text = "Draw can be claimed."
        else:
            check_text = ""
            
        if check_text:
            check_surf, check_rect = font.render(check_text, (255, 0, 0))
            self.screen.blit(check_surf, (BOARD_SIZE + 20, 60))
        
        # Model status
        model_surf, model_rect = small_font.render(self.model_status, TEXT_COLOR)
        self.screen.blit(model_surf, (BOARD_SIZE + 20, 100))
        
        # User message
        message_surf, message_rect = small_font.render(self.message, TEXT_COLOR)
        self.screen.blit(message_surf, (BOARD_SIZE + 20, 140))
        
        # Draw W/B buttons if color not selected
        if self.player_color is None:
            self.draw_button("W", BOARD_SIZE + 50, 200, 60, 40)
            self.draw_button("B", BOARD_SIZE + 150, 200, 60, 40)
    
    def draw_button(self, text, x, y, width, height):
        """Draw a button on the screen."""
        mouse_pos = pygame.mouse.get_pos()
        button_rect = pygame.Rect(x, y, width, height)
        
        # Check if mouse is hovering over button
        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, BUTTON_HOVER_COLOR, button_rect)
        else:
            pygame.draw.rect(self.screen, BUTTON_COLOR, button_rect)
        
        # Button text
        text_surf, text_rect = font.render(text, BUTTON_TEXT_COLOR)
        text_rect.center = button_rect.center
        self.screen.blit(text_surf, text_rect)
        
        return button_rect
    
    def get_square_from_pos(self, pos):
        """Convert mouse position to chess square."""
        x, y = pos
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return None
            
        col = x // SQUARE_SIZE
        row = 7 - (y // SQUARE_SIZE)  # Invert row since chess board is bottom-up
        return chess.square(col, row)
    
    def get_highlighted_moves(self, square):
        """Get all legal moves from the selected square."""
        moves = []
        for move in self.board.legal_moves:
            if move.from_square == square:
                moves.append(move.to_square)
        return moves
    
    def make_human_move(self, from_square, to_square):
        """Make a human move on the board."""
        move = chess.Move(from_square, to_square)
        
        # Check for promotion
        if (self.board.piece_at(from_square).piece_type == chess.PAWN and 
            (to_square // 8 == 7 or to_square // 8 == 0)):
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        
        if move in self.board.legal_moves:
            self.board.push(move)
            self.last_move = move
            self.message = f"Made move: {move.uci()}"
            return True
        else:
            self.message = "Illegal move!"
            return False
    
    def make_model_move(self):
        """Make a move using the neural network model."""
        if self.board.is_game_over():
            return
            
        if not self.model:
            self.load_model()
            
        try:
            fen = self.board.fen()
            start_time = time.time()
            move_uci = self.model.predict_move(fen)
            elapsed_time = time.time() - start_time
            
            if move_uci:
                move = chess.Move.from_uci(move_uci)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.last_move = move
                    self.message = f"Model move: {move_uci} ({elapsed_time:.2f}s)"
                else:
                    self.message = f"Model returned illegal move: {move_uci}"
            else:
                self.message = "Model couldn't find a move"
                
        except Exception as e:
            self.message = f"Error in model: {str(e)[:50]}..."
    
    def handle_click(self, pos):
        """Handle mouse click on the board."""
        # Check if game is over
        if self.board.is_game_over():
            return
            
        # Handle button clicks if color not selected
        if self.player_color is None:
            button_w = pygame.Rect(BOARD_SIZE + 50, 200, 60, 40)
            button_b = pygame.Rect(BOARD_SIZE + 150, 200, 60, 40)
            
            if button_w.collidepoint(pos):
                self.player_color = chess.WHITE
                self.message = "Your model plays as White. Waiting for White's move."
                self.load_model()
                # If model is White, it goes first
                if self.board.turn == chess.WHITE:
                    self.make_model_move()
                return
                
            elif button_b.collidepoint(pos):
                self.player_color = chess.BLACK
                self.message = "Your model plays as Black. Waiting for White's move."
                self.load_model()
                return
                
            # Not a button click, ignore
            return
        
        # Check if it's the model's turn
        if (self.board.turn == chess.WHITE and self.player_color == chess.WHITE) or \
           (self.board.turn == chess.BLACK and self.player_color == chess.BLACK):
            self.make_model_move()
            return
            
        # Handle board clicks for human moves
        square = self.get_square_from_pos(pos)
        if square is None:
            return
            
        # If no square is selected, select this square if it has a piece of the correct color
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and ((piece.color == chess.WHITE and self.board.turn == chess.WHITE) or 
                         (piece.color == chess.BLACK and self.board.turn == chess.BLACK)):
                self.selected_square = square
                self.highlighted_squares = self.get_highlighted_moves(square)
            return
            
        # If a square is already selected
        if square in self.highlighted_squares:
            # Make the move
            self.make_human_move(self.selected_square, square)
            
            # Reset selection
            self.selected_square = None
            self.highlighted_squares = []
            
            # If move was successful and it's the model's turn, make a model move
            if (self.board.turn == chess.WHITE and self.player_color == chess.WHITE) or \
               (self.board.turn == chess.BLACK and self.player_color == chess.BLACK):
                self.make_model_move()
        else:
            # Deselect or select a new piece
            piece = self.board.piece_at(square)
            if piece and ((piece.color == chess.WHITE and self.board.turn == chess.WHITE) or 
                         (piece.color == chess.BLACK and self.board.turn == chess.BLACK)):
                self.selected_square = square
                self.highlighted_squares = self.get_highlighted_moves(square)
            else:
                self.selected_square = None
                self.highlighted_squares = []
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:  # Left click
                    self.handle_click(event.pos)
            
            # Draw everything
            self.screen.fill((30, 30, 30))
            self.draw_board()
            self.draw_pieces()
            self.draw_status()
            
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()

def main():
    """Main function to start the chess game."""
    try:
        game = ChessGame()
        game.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
