"""
Chess AI Interface
Just a GUI I threw together for playing chess against my AI.
Works pretty well but could probably use some cleanup.
"""

import chess
import pygame
import sys
import json
from pathlib import Path
from logic import (
    ChessAI, ValueNetwork, SearchConfig, LearningConfig,
    visualize_tree, OpeningBook
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Colors and stuff
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (130, 151, 105)
MOVE_HINT = (205, 210, 106)
BACKGROUND = (30, 30, 30)
TEXT_COLOR = (240, 240, 240)
PANEL_BG = (45, 45, 45)
ACCENT = (100, 150, 230)

SQUARE_SIZE = 80
BOARD_SIZE = 640
PANEL_WIDTH = 300
MARGIN = 20
FPS = 60


class PieceRenderer:
    """Loads and draws pieces"""
    
    def __init__(self, square_size, pieces_dir="pieces"):
        self.square_size = square_size
        self.pieces_dir = Path(pieces_dir)
        self.pieces = {}
        self.load_pieces()
    
    def load_pieces(self):
        # piece file mapping
        piece_files = {
            'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
            'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'
        }
        
        for symbol, filename in piece_files.items():
            try:
                img_path = self.pieces_dir / f"{filename}.png"
                img = pygame.image.load(str(img_path))
                self.pieces[symbol] = pygame.transform.smoothscale(
                    img, (self.square_size, self.square_size)
                )
            except Exception as e:
                logger.error(f"Couldn't load {filename}: {e}")
                # fallback to placeholder
                surf = pygame.Surface((self.square_size, self.square_size))
                surf.fill((200, 200, 200))
                pygame.draw.circle(surf, (100, 100, 100), 
                         (self.square_size // 2, self.square_size // 2),
                         self.square_size // 3)
                self.pieces[symbol] = surf
    
    def get_piece(self, symbol):
        return self.pieces.get(symbol)


class ChessGUI:
    """Main game interface"""
    
    def __init__(self, screen):
        self.screen = screen
        self.piece_renderer = PieceRenderer(SQUARE_SIZE)
        
        # setup fonts
        self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_normal = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Arial", 14)
        
        # game state
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        self.hint_move = None
        self.last_move = None
        
        # mode stuff
        self.player_color = None
        self.is_self_play = False
        
        # stats tracking
        self.move_count = 0
        self.game_history = []
        self.last_search_stats = {}
        self.games_completed = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
    
    def get_board_position(self):
        """figure out where to draw the board"""
        w = self.screen.get_width()
        h = self.screen.get_height()
        x = (w - PANEL_WIDTH - BOARD_SIZE) // 2
        y = (h - BOARD_SIZE) // 2
        return x, y
    
    def screen_to_square(self, pos):
        """convert click to chess square"""
        x, y = pos
        bx, by = self.get_board_position()
        
        file = (x - bx) // SQUARE_SIZE
        rank = (y - by) // SQUARE_SIZE
        
        if 0 <= file < 8 and 0 <= rank < 8:
            return chess.square(file, 7 - rank)
        return None
    
    def draw_board(self):
        bx, by = self.get_board_position()
        
        # draw all squares
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank)
                rect = pygame.Rect(
                    bx + file * SQUARE_SIZE,
                    by + rank * SQUARE_SIZE,
                    SQUARE_SIZE, SQUARE_SIZE
                )
                
                # color the square
                light = (rank + file) % 2 == 0
                color = LIGHT_SQUARE if light else DARK_SQUARE
                
                if sq == self.selected_square:
                    color = HIGHLIGHT
                
                # highlight last move
                if self.last_move and sq in [self.last_move.from_square, self.last_move.to_square]:
                    color = tuple(max(0, c - 30) for c in color)
                
                pygame.draw.rect(self.screen, color, rect)
                
                # draw piece if there is one
                piece = self.board.piece_at(sq)
                if piece:
                    piece_surf = self.piece_renderer.get_piece(piece.symbol())
                    if piece_surf:
                        self.screen.blit(piece_surf, rect)
        
        # show possible moves
        for move in self.valid_moves:
            f = chess.square_file(move.to_square)
            r = 7 - chess.square_rank(move.to_square)
            cx = bx + f * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = by + r * SQUARE_SIZE + SQUARE_SIZE // 2
            
            pygame.draw.circle(self.screen, MOVE_HINT, (cx, cy), 12)
            pygame.draw.circle(self.screen, DARK_SQUARE, (cx, cy), 12, 2)
        
        # hint arrow
        if self.hint_move:
            self.draw_arrow(bx, by, self.hint_move)
        
        # board labels
        self.draw_labels(bx, by)
    
    def draw_arrow(self, bx, by, move):
        from_f = chess.square_file(move.from_square)
        from_r = 7 - chess.square_rank(move.from_square)
        to_f = chess.square_file(move.to_square)
        to_r = 7 - chess.square_rank(move.to_square)
        
        sx = bx + from_f * SQUARE_SIZE + SQUARE_SIZE // 2
        sy = by + from_r * SQUARE_SIZE + SQUARE_SIZE // 2
        ex = bx + to_f * SQUARE_SIZE + SQUARE_SIZE // 2
        ey = by + to_r * SQUARE_SIZE + SQUARE_SIZE // 2
        
        pygame.draw.line(self.screen, ACCENT, (sx, sy), (ex, ey), 5)
        pygame.draw.circle(self.screen, ACCENT, (ex, ey), 10)
    
    def draw_labels(self, bx, by):
        for i in range(8):
            # files
            label = self.font_small.render(chr(ord('a') + i), True, TEXT_COLOR)
            x = bx + i * SQUARE_SIZE + SQUARE_SIZE // 2 - 5
            y = by + BOARD_SIZE + 5
            self.screen.blit(label, (x, y))
            
            # ranks
            label = self.font_small.render(str(8 - i), True, TEXT_COLOR)
            x = bx - 20
            y = by + i * SQUARE_SIZE + SQUARE_SIZE // 2 - 8
            self.screen.blit(label, (x, y))
    
    def draw_side_panel(self):
        px = self.screen.get_width() - PANEL_WIDTH
        panel_rect = pygame.Rect(px, 0, PANEL_WIDTH, self.screen.get_height())
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        
        y = MARGIN
        
        # title
        title = self.font_title.render("Chess AI", True, ACCENT)
        self.screen.blit(title, (px + MARGIN, y))
        y += 50
        
        # current mode
        mode = "Self-Play" if self.is_self_play else "Player vs AI"
        text = self.font_normal.render(f"Mode: {mode}", True, TEXT_COLOR)
        self.screen.blit(text, (px + MARGIN, y))
        y += 30
        
        # whose turn
        turn = "White" if self.board.turn else "Black"
        text = self.font_normal.render(f"Turn: {turn}", True, TEXT_COLOR)
        self.screen.blit(text, (px + MARGIN, y))
        y += 30
        
        # move counter
        text = self.font_normal.render(f"Move: {self.move_count}", True, TEXT_COLOR)
        self.screen.blit(text, (px + MARGIN, y))
        y += 30
        
        # self play stats
        if self.is_self_play:
            stats = [
                f"Games: {self.games_completed}",
                f"White: {self.white_wins}",
                f"Black: {self.black_wins}",
                f"Draws: {self.draws}"
            ]
            for s in stats:
                text = self.font_small.render(s, True, TEXT_COLOR)
                self.screen.blit(text, (px + MARGIN, y))
                y += 20
        
        y += 20
        
        # search stats from last move
        if self.last_search_stats and not self.last_search_stats.get('from_opening_book'):
            title = self.font_normal.render("Last Move:", True, ACCENT)
            self.screen.blit(title, (px + MARGIN, y))
            y += 25
            
            stats = [
                f"Score: {self.last_search_stats.get('best_score', 0):.1f}",
                f"Nodes: {self.last_search_stats.get('nodes_searched', 0):,}",
                f"Time: {self.last_search_stats.get('time', 0):.2f}s",
                f"NPS: {self.last_search_stats.get('nps', 0):.0f}",
                f"TT Hit: {self.last_search_stats.get('tt_hit_rate', 0):.1%}"
            ]
            
            for s in stats:
                text = self.font_small.render(s, True, TEXT_COLOR)
                self.screen.blit(text, (px + MARGIN + 10, y))
                y += 20
        elif self.last_search_stats and self.last_search_stats.get('from_opening_book'):
            text = self.font_small.render("(Opening Book)", True, ACCENT)
            self.screen.blit(text, (px + MARGIN, y))
            y += 20
        
        y += 30
        
        # controls help
        title = self.font_normal.render("Controls:", True, ACCENT)
        self.screen.blit(title, (px + MARGIN, y))
        y += 25
        
        controls = [
            "H - Get hint",
            "V - Visualize search tree",
            "R - Reset game",
            "N - Show network stats",
            "ESC - Quit"
        ]
        
        for c in controls:
            text = self.font_small.render(c, True, TEXT_COLOR)
            self.screen.blit(text, (px + MARGIN + 10, y))
            y += 20
    
    def render(self):
        self.screen.fill(BACKGROUND)
        self.draw_board()
        self.draw_side_panel()
        pygame.display.flip()


def show_promotion_dialog(screen, color):
    """show dialog to pick promotion piece"""
    font = pygame.font.SysFont("Arial", 24)
    
    # promotion choices
    pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    piece_names = ["Queen", "Rook", "Bishop", "Knight"]
    
    # create semi-transparent overlay
    overlay = pygame.Surface(screen.get_size())
    overlay.set_alpha(200)
    overlay.fill((20, 20, 20))
    
    selected = 0
    
    while True:
        screen.blit(overlay, (0, 0))
        
        # title
        title = font.render("Choose Promotion:", True, TEXT_COLOR)
        rect = title.get_rect(center=(screen.get_width() // 2, 200))
        screen.blit(title, rect)
        
        # options
        y = 260
        for i, name in enumerate(piece_names):
            color_text = ACCENT if i == selected else TEXT_COLOR
            text = font.render(name, True, color_text)
            rect = text.get_rect(center=(screen.get_width() // 2, y + i * 50))
            screen.blit(text, rect)
            
            if i == selected:
                pygame.draw.rect(screen, ACCENT, rect.inflate(20, 10), 2, border_radius=5)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(pieces)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(pieces)
                elif event.key == pygame.K_RETURN:
                    return pieces[selected]
                elif event.key == pygame.K_ESCAPE:
                    return chess.QUEEN  # default to queen if they cancel
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # also allow clicking
                mx, my = pygame.mouse.get_pos()
                base_y = 260
                for i in range(len(pieces)):
                    if abs(my - (base_y + i * 50)) < 25:
                        return pieces[i]


def show_menu(screen):
    """Main menu for game mode selection"""
    font_title = pygame.font.SysFont("Arial", 48, bold=True)
    font_option = pygame.font.SysFont("Arial", 28)
    font_small = pygame.font.SysFont("Arial", 18)
    
    selected = 0
    options = [
        ("Play as White", chess.WHITE, False),
        ("Play as Black", chess.BLACK, False),
        ("Watch Self-Play", None, True)
    ]
    
    while True:
        screen.fill(BACKGROUND)
        
        title = font_title.render("Chess AI", True, ACCENT)
        rect = title.get_rect(center=(screen.get_width() // 2, 100))
        screen.blit(title, rect)
        
        subtitle = font_small.render("Classical + RL Hybrid", True, TEXT_COLOR)
        rect = subtitle.get_rect(center=(screen.get_width() // 2, 150))
        screen.blit(subtitle, rect)
        
        # menu options
        y = 250
        for i, (text, _, _) in enumerate(options):
            color = ACCENT if i == selected else TEXT_COLOR
            opt = font_option.render(text, True, color)
            rect = opt.get_rect(center=(screen.get_width() // 2, y + i * 60))
            screen.blit(opt, rect)
            
            if i == selected:
                pygame.draw.rect(screen, ACCENT, rect.inflate(20, 10), 2, border_radius=5)
        
        instructions = font_option.render("↑↓ select, ENTER confirm", True, TEXT_COLOR)
        rect = instructions.get_rect(center=(screen.get_width() // 2, screen.get_height() - 50))
        screen.blit(instructions, rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    _, color, selfplay = options[selected]
                    return color, selfplay
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()


def main():
    pygame.init()
    screen = pygame.display.set_mode((1100, 700), pygame.RESIZABLE)
    pygame.display.set_caption("Chess AI")
    clock = pygame.time.Clock()
    
    # get game mode
    player_color, is_self_play = show_menu(screen)
    
    # setup AI
    search_cfg = SearchConfig(max_depth=4, time_limit=1.0)
    learn_cfg = LearningConfig(
        learning_rate=0.002,
        classical_blend=0.8,
        exploration_noise=0.1
    )
    
    # separate networks for self-play
    white_net = ValueNetwork(learn_cfg, 'white_hybrid.pkl')
    black_net = ValueNetwork(learn_cfg, 'black_hybrid.pkl') if is_self_play else white_net
    
    white_ai = ChessAI(white_net, search_cfg, learn_cfg)
    black_ai = ChessAI(black_net, search_cfg, learn_cfg) if is_self_play else white_ai
    
    # init gui
    gui = ChessGUI(screen)
    gui.player_color = player_color
    gui.is_self_play = is_self_play
    
    running = True
    ai_thinking = False
    
    logger.info(f"Game started - {'Self-Play' if is_self_play else 'Player vs AI'}")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
                elif event.key == pygame.K_r:
                    gui.board.reset()
                    gui.game_history.clear()
                    gui.move_count = 0
                    gui.selected_square = None
                    gui.valid_moves = []
                    gui.hint_move = None
                    gui.last_move = None
                    logger.info("Reset")
                    
                elif event.key == pygame.K_h and not is_self_play:
                    if gui.board.turn == player_color:
                        ai = white_ai if player_color == chess.WHITE else black_ai
                        gui.hint_move, gui.last_search_stats = ai.compute_best_move(gui.board, use_opening_book=False)
                        logger.info(f"Hint: {gui.hint_move.uci() if gui.hint_move else 'None'}")
                        
                elif event.key == pygame.K_v and gui.last_search_stats:
                    visualize_tree(gui.last_search_stats, 'search_tree.png')
                    logger.info("Tree viz saved")
                
                elif event.key == pygame.K_n:
                    stats = white_net.get_statistics()
                    logger.info(f"White: {json.dumps(stats, indent=2)}")
                    if is_self_play:
                        stats = black_net.get_statistics()
                        logger.info(f"Black: {json.dumps(stats, indent=2)}")
                    
            elif event.type == pygame.MOUSEBUTTONDOWN and not is_self_play:
                if gui.board.turn == player_color and not ai_thinking:
                    sq = gui.screen_to_square(pygame.mouse.get_pos())
                    
                    if sq is not None:
                        if gui.selected_square is None:
                            piece = gui.board.piece_at(sq)
                            if piece and piece.color == player_color:
                                gui.selected_square = sq
                                gui.valid_moves = [m for m in gui.board.legal_moves 
                                                 if m.from_square == sq]
                        else:
                            # check if this is a pawn promotion
                            piece = gui.board.piece_at(gui.selected_square)
                            if piece and piece.piece_type == chess.PAWN and chess.square_rank(sq) in [0, 7]:
                                # show promotion menu
                                promo_piece = show_promotion_dialog(screen, player_color)
                                if promo_piece:
                                    move = chess.Move(gui.selected_square, sq, promotion=promo_piece)
                                    if move in gui.board.legal_moves:
                                        gui.board.push(move)
                                        gui.game_history.append((gui.board.fen(), move.uci()))
                                        gui.move_count += 1
                                        gui.last_move = move
                                        gui.hint_move = None
                                        ai_thinking = True
                            else:
                                move = chess.Move(gui.selected_square, sq)
                                if move in gui.board.legal_moves:
                                    gui.board.push(move)
                                    gui.game_history.append((gui.board.fen(), move.uci()))
                                    gui.move_count += 1
                                    gui.last_move = move
                                    gui.hint_move = None
                                    ai_thinking = True
                            
                            gui.selected_square = None
                            gui.valid_moves = []
        
        # AI makes move
        if not gui.board.is_game_over():
            should_move = is_self_play or (not is_self_play and gui.board.turn != player_color)
            
            if should_move and not ai_thinking:
                ai_thinking = True
            
            if ai_thinking:
                ai = white_ai if gui.board.turn == chess.WHITE else black_ai
                move, stats = ai.compute_best_move(gui.board, use_opening_book=True)
                
                if move:
                    gui.board.push(move)
                    gui.game_history.append((gui.board.fen(), move.uci()))
                    gui.move_count += 1
                    gui.last_move = move
                    gui.last_search_stats = stats
                    gui.hint_move = None
                
                ai_thinking = False
                
                if is_self_play:
                    pygame.time.delay(300)  # slow down so we can watch
        
        # game over handling
        if gui.board.is_game_over():
            result = gui.board.result()
            logger.info(f"Game over: {result}")
            
            gui.games_completed += 1
            
            if result == "1-0":
                reward = 1.0
                gui.white_wins += 1
                logger.info("White wins")
            elif result == "0-1":
                reward = -1.0
                gui.black_wins += 1
                logger.info("Black wins")
            else:
                reward = 0.0
                gui.draws += 1
                logger.info("Draw")
            
            # update networks
            logger.info("Updating networks...")
            white_ai.update_from_game(gui.game_history, reward)
            if is_self_play:
                black_ai.update_from_game(gui.game_history, -reward)
            
            white_net.save()
            if is_self_play:
                black_net.save()
            
            stats = white_net.get_statistics()
            logger.info(f"White: {stats['num_positions']} pos, noise: {stats['exploration_noise']:.4f}")
            if is_self_play:
                stats = black_net.get_statistics()
                logger.info(f"Black: {stats['num_positions']} pos, noise: {stats['exploration_noise']:.4f}")
            
            # start new game
            gui.board.reset()
            gui.game_history.clear()
            gui.move_count = 0
            gui.last_move = None
            
            pygame.time.delay(2000)
        
        gui.render()
        clock.tick(FPS)
    
    # save before exit
    white_net.save()
    if is_self_play:
        black_net.save()
    
    pygame.quit()
    logger.info("Done")


if __name__ == "__main__":
    main()