"""
Chess AI Engine - hybrid classical + RL approach
Trying to combine good old fashioned alpha-beta with some learning.
Still experimental but seems to work ok.
"""
import chess
import time
import pickle
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque, defaultdict
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    max_depth: int = 4
    time_limit: float = 1.0
    aspiration_window: int = 50
    null_move_reduction: int = 2
    use_transposition_table: bool = True
    
@dataclass
class LearningConfig:
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    replay_buffer_size: int = 10000
    batch_size: int = 32
    exploration_noise: float = 0.1
    noise_decay: float = 0.9995
    min_noise: float = 0.01
    classical_blend: float = 0.8

class OpeningBook:
    """Some basic openings to avoid the same games every time"""
    OPENINGS = [
        ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4'],  # italian
        ['d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3'],  # QG
        ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4'],  # sicilian
        ['d2d4', 'g8f6', 'c2c4', 'g7g6', 'b1c3'],  # KID
        ['g1f3', 'd7d5', 'c2c4', 'c7c6', 'e2e3'],  # reti
        ['e2e4', 'e7e6', 'd2d4', 'd7d5', 'b1c3'],  # french
        ['d2d4', 'g8f6', 'g1f3', 'e7e6', 'c2c4'],  # nimzo
    ]
    
    @staticmethod
    def get_opening_move(board, move_num):
        if move_num >= 5:
            return None
        
        opening = OpeningBook.OPENINGS[np.random.randint(0, len(OpeningBook.OPENINGS))]
        
        if move_num < len(opening):
            try:
                mv = chess.Move.from_uci(opening[move_num])
                if mv in board.legal_moves:
                    return mv
            except:
                pass
        return None

class ExperienceReplay:
    """replay buffer for more stable learning"""
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def __len__(self):
        return len(self.buffer)

class TranspositionTable:
    """cache for positions we've already evaluated"""
    
    def __init__(self, max_size=1000000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def store(self, zobrist, depth, value, flag, best_move=None):
        if len(self.table) >= self.max_size:
            # just remove first entry when full
            self.table.pop(next(iter(self.table)))
            
        self.table[zobrist] = {
            'depth': depth,
            'value': value,
            'flag': flag,  # exact, lowerbound, upperbound
            'best_move': best_move
        }
        
    def probe(self, zobrist, depth, alpha, beta):
        if zobrist not in self.table:
            self.misses += 1
            return None
            
        entry = self.table[zobrist]
        if entry['depth'] >= depth:
            self.hits += 1
            flag = entry['flag']
            val = entry['value']
            
            if flag == 'exact':
                return val, entry['best_move']
            elif flag == 'lowerbound' and val >= beta:
                return val, entry['best_move']
            elif flag == 'upperbound' and val <= alpha:
                return val, entry['best_move']
                
        self.misses += 1
        return None
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class ValueNetwork:
    """learned position values to augment classical eval"""
    
    def __init__(self, config, filepath='value_network.pkl'):
        self.config = config
        self.filepath = Path(filepath)
        self.position_values = defaultdict(float)
        self.visit_counts = defaultdict(int)
        self.exploration_noise = config.exploration_noise
        self.games_played = 0
        self.load_if_exists()
        
    def load_if_exists(self):
        try:
            if self.filepath.exists():
                with open(self.filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.position_values = defaultdict(float, data.get('position_values', {}))
                    self.visit_counts = defaultdict(int, data.get('visit_counts', {}))
                    self.exploration_noise = data.get('exploration_noise', self.config.exploration_noise)
                    self.games_played = data.get('games_played', 0)
                logger.info(f"Loaded network: {len(self.position_values)} positions, {self.games_played} games")
        except Exception as e:
            logger.error(f"Load failed: {e}")
            
    def save(self):
        try:
            data = {
                'position_values': dict(self.position_values),
                'visit_counts': dict(self.visit_counts),
                'exploration_noise': self.exploration_noise,
                'games_played': self.games_played,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_positions': len(self.position_values)
                }
            }
            with open(self.filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved to {self.filepath}")
        except Exception as e:
            logger.error(f"Save failed: {e}")
            
    def get_position_value(self, fen):
        return self.position_values[fen]
    
    def get_confidence(self, fen):
        """confidence based on how many times we've seen this position"""
        visits = self.visit_counts[fen]
        return min(1.0, visits / 1000.0)
    
    def update(self, fen, target):
        """TD learning update"""
        current = self.position_values[fen]
        visits = self.visit_counts[fen]
        
        # adaptive learning rate
        lr = self.config.learning_rate / np.sqrt(1 + visits)
        
        td_error = target - current
        self.position_values[fen] += lr * td_error
        self.visit_counts[fen] += 1
    
    def decay_noise(self):
        self.exploration_noise = max(self.config.min_noise,
                                    self.exploration_noise * self.config.noise_decay)
        
    def get_statistics(self):
        vals = list(self.position_values.values())
        return {
            'num_positions': len(self.position_values),
            'avg_value': np.mean(vals) if vals else 0.0,
            'std_value': np.std(vals) if vals else 0.0,
            'exploration_noise': self.exploration_noise,
            'max_visits': max(self.visit_counts.values()) if self.visit_counts else 0,
            'games_played': self.games_played
        }

class ChessEvaluator:
    """board evaluation with piece square tables"""
    
    # piece square tables
    PAWN_TABLE = np.array([
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ])
    
    KNIGHT_TABLE = np.array([
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ])
    
    BISHOP_TABLE = np.array([
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ])
    
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    @staticmethod
    def evaluate(board):
        """evaluate position from current player perspective"""
        if board.is_checkmate():
            return -30000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0.0
        
        # material + position
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                val = ChessEvaluator.PIECE_VALUES[piece.piece_type]
                
                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                
                eval_rank = rank if piece.color == chess.WHITE else 7 - rank
                
                # positional bonus
                if piece.piece_type == chess.PAWN:
                    pos_val = ChessEvaluator.PAWN_TABLE[eval_rank][file]
                elif piece.piece_type == chess.KNIGHT:
                    pos_val = ChessEvaluator.KNIGHT_TABLE[eval_rank][file]
                elif piece.piece_type == chess.BISHOP:
                    pos_val = ChessEvaluator.BISHOP_TABLE[eval_rank][file]
                else:
                    pos_val = 0
                
                total = val + pos_val
                score += total if piece.color == chess.WHITE else -total
        
        # mobility bonus
        mobility = len(list(board.legal_moves))
        score += mobility * 10 if board.turn == chess.WHITE else -mobility * 10
        
        # king safety
        score += ChessEvaluator.eval_king_safety(board, chess.WHITE)
        score -= ChessEvaluator.eval_king_safety(board, chess.BLACK)
        
        # pawn structure
        score += ChessEvaluator.eval_pawns(board, chess.WHITE)
        score -= ChessEvaluator.eval_pawns(board, chess.BLACK)
        
        # flip for black
        if board.turn == chess.BLACK:
            score = -score
        
        return score
    
    @staticmethod
    def eval_king_safety(board, color):
        king_sq = board.king(color)
        if king_sq is None:
            return 0.0
            
        safety = 0.0
        
        # middlegame king safety
        if len(board.piece_map()) > 10:
            kf = chess.square_file(king_sq)
            kr = chess.square_rank(king_sq)
            
            # prefer corners in middlegame
            if (kf in [0, 1, 6, 7]) and (kr in [0, 1, 6, 7]):
                safety += 20
                
        return safety
    
    @staticmethod
    def eval_pawns(board, color):
        """pawn structure evaluation"""
        score = 0.0
        pawns = board.pieces(chess.PAWN, color)
        
        pawn_files = defaultdict(int)
        for sq in pawns:
            pawn_files[chess.square_file(sq)] += 1
        
        # doubled pawns bad
        for f, cnt in pawn_files.items():
            if cnt > 1:
                score -= 10 * (cnt - 1)
        
        # isolated pawns bad
        for f in pawn_files:
            if (f - 1) not in pawn_files and (f + 1) not in pawn_files:
                score -= 15
                
        return score
    
    @staticmethod
    def material_count(board):
        """total material (white perspective)"""
        total = 0
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            total += len(board.pieces(pt, chess.WHITE)) * ChessEvaluator.PIECE_VALUES[pt]
            total -= len(board.pieces(pt, chess.BLACK)) * ChessEvaluator.PIECE_VALUES[pt]
        return total

class ChessAI:
    """main AI engine"""
    
    def __init__(self, value_network, search_config=SearchConfig(), learning_config=LearningConfig()):
        self.value_network = value_network
        self.search_config = search_config
        self.learning_config = learning_config
        self.tt = TranspositionTable()
        self.replay = ExperienceReplay(learning_config.replay_buffer_size)
        self.nodes = 0
        self.start_time = 0.0
        
    def compute_best_move(self, board, use_opening_book=True):
        """find best move with iterative deepening"""
        
        # opening book first
        if use_opening_book and board.fullmove_number <= 5:
            book_move = OpeningBook.get_opening_move(board, board.fullmove_number - 1)
            if book_move:
                logger.info(f"Book move: {book_move.uci()}")
                return book_move, {'from_opening_book': True, 'move': book_move.uci()}
        
        self.nodes = 0
        self.start_time = time.time()
        
        best_move = None
        best_score = -float('inf')
        stats = {'depths': {}}
        
        legal = list(board.legal_moves)
        if not legal:
            return None, stats
        
        ordered = self.order_moves(board, legal)
        
        alpha = -float('inf')
        beta = float('inf')
        
        # iterative deepening
        for depth in range(1, self.search_config.max_depth + 1):
            if time.time() - self.start_time > self.search_config.time_limit:
                break
            
            # aspiration windows for deeper searches
            if depth > 2 and best_score != -float('inf'):
                alpha = best_score - self.search_config.aspiration_window
                beta = best_score + self.search_config.aspiration_window
            else:
                alpha = -float('inf')
                beta = float('inf')
            
            depth_best = None
            depth_score = -float('inf')
            depth_stats = {}
            
            for move in ordered:
                board.push(move)
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, False)
                board.pop()
                
                depth_stats[move.uci()] = score
                
                if score > depth_score:
                    depth_score = score
                    depth_best = move
                    
                alpha = max(alpha, score)
                
                if time.time() - self.start_time > self.search_config.time_limit:
                    break
            
            if depth_best:
                best_move = depth_best
                best_score = depth_score
                # reorder for next iteration
                ordered = [best_move] + [m for m in ordered if m != best_move]
                
            stats['depths'][depth] = depth_stats
            
        elapsed = time.time() - self.start_time
        stats.update({
            'best_move': best_move.uci() if best_move else None,
            'best_score': best_score,
            'nodes_searched': self.nodes,
            'time': elapsed,
            'nps': self.nodes / elapsed if elapsed > 0 else 0,
            'tt_hit_rate': self.tt.get_hit_rate()
        })
        
        logger.info(f"Move: {best_move.uci() if best_move else 'None'}, "
                   f"Score: {best_score:.2f}, Nodes: {self.nodes}, "
                   f"Time: {elapsed:.2f}s")
        
        return best_move, stats
    
    def order_moves(self, board, moves):
        """order moves with heuristics"""
        scored = []
        
        for move in moves:
            score = 0.0
            
            # captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # MVV-LVA
                    score += (ChessEvaluator.PIECE_VALUES[victim.piece_type] -
                            ChessEvaluator.PIECE_VALUES[attacker.piece_type] / 10)
            
            # checks
            if board.gives_check(move):
                score += 50
            
            # center control
            to_sq = move.to_square
            tf = chess.square_file(to_sq)
            tr = chess.square_rank(to_sq)
            if 2 <= tf <= 5 and 2 <= tr <= 5:
                score += 10
            
            scored.append((move, score))
        
        return [m for m, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    
    def alpha_beta(self, board, depth, alpha, beta, maximizing):
        """alpha beta search"""
        self.nodes += 1
        
        # time check
        if time.time() - self.start_time > self.search_config.time_limit:
            return self.eval_hybrid(board)
        
        # TT probe
        zobrist = hash(board.fen())
        tt_result = self.tt.probe(zobrist, depth, alpha, beta)
        if tt_result:
            return tt_result[0]
        
        # terminal or depth limit
        if depth <= 0 or board.is_game_over():
            score = self.quiescence(board, alpha, beta, 3)
            self.tt.store(zobrist, depth, score, 'exact')
            return score
        
        legal = list(board.legal_moves)
        if not legal:
            return self.eval_hybrid(board)
        
        ordered = self.order_moves(board, legal)
        
        best_score = -float('inf') if maximizing else float('inf')
        best_move = None
        flag = 'upperbound'
        
        for move in ordered:
            board.push(move)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha, not maximizing)
            board.pop()
            
            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            
            if alpha >= beta:
                flag = 'lowerbound' if maximizing else 'upperbound'
                break
        
        if alpha < best_score < beta:
            flag = 'exact'
        
        self.tt.store(zobrist, depth, best_score, flag, best_move)
        return best_score
    
    def quiescence(self, board, alpha, beta, depth):
        """quiescence search for captures"""
        stand_pat = self.eval_hybrid(board)
        
        if depth <= 0:
            return stand_pat
        
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        
        for move in captures:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def eval_hybrid(self, board):
        """blend classical and learned evaluation"""
        classical = ChessEvaluator.evaluate(board)
        
        fen = board.fen()
        learned = self.value_network.get_position_value(fen)
        confidence = self.value_network.get_confidence(fen)
        
        # blend based on confidence
        blend = confidence * (1 - self.learning_config.classical_blend)
        final = classical * (1 - blend) + learned * blend
        
        return final
    
    def update_from_game(self, game_history, result):
        """update network from completed game
        result: 1.0 = win, 0.0 = draw, -1.0 = loss"""
        
        n = len(game_history)
        if n == 0:
            return
        
        fens = [pair[0] for pair in game_history]
        moves = [pair[1] for pair in game_history]
        
        # track material and turns
        material = []
        turns = []
        for fen in fens:
            b = chess.Board(fen)
            material.append(ChessEvaluator.material_count(b))
            turns.append(b.turn)
        
        # terminal reward from last player perspective
        last_white = turns[-1]
        terminal = result if last_white else -result
        
        # add to replay buffer
        for i in range(n):
            fen = fens[i]
            move = moves[i]
            turn = turns[i]
            done = (i == n - 1)
            
            if done:
                reward = terminal * 10000
                next_fen = ''
            else:
                mat_change = material[i + 1] - material[i]
                reward = (mat_change if turn else -mat_change) * 0.1
                next_fen = fens[i + 1]
            
            self.replay.add(fen, move, reward, next_fen, done)
        
        # train from replay
        for _ in range(5):
            if len(self.replay) < self.learning_config.batch_size:
                break
            
            batch = self.replay.sample(self.learning_config.batch_size)
            for state, _, reward, next_state, done in batch:
                if done:
                    target = reward
                else:
                    next_val = self.value_network.get_position_value(next_state)
                    target = reward + self.learning_config.discount_factor * (-next_val)
                
                self.value_network.update(state, target)
        
        self.value_network.decay_noise()
        self.value_network.games_played += 1

def visualize_tree(stats, output='search_tree.png'):
    """make a picture of the search tree"""
    if not stats.get('depths'):
        logger.warning("No tree to visualize")
        return
    
    try:
        G = nx.DiGraph()
        labels = {}
        colors = []
        
        G.add_node("root")
        labels["root"] = "Root"
        colors.append("lightblue")
        
        node_id = 0
        for depth, moves in stats['depths'].items():
            if not moves:
                continue
            best = max(moves.items(), key=lambda x: x[1])
            for move, score in moves.items():
                node_id += 1
                name = f"d{depth}_{node_id}"
                G.add_node(name)
                labels[name] = f"{move}\n{score:.1f}"
                color = "gold" if move == best[0] else "lightblue"
                colors.append(color)
                G.add_edge("root", name)
        
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw(G, pos, labels=labels, node_color=colors,
               node_size=2000, font_size=8, font_weight='bold',
               edge_color='gray', arrows=True, arrowsize=20)
        plt.title("Search Tree", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved to {output}")
    except Exception as e:
        logger.error(f"Viz error: {e}")

if __name__ == "__main__":
    # quick test
    cfg = SearchConfig(max_depth=4, time_limit=1.0)
    learn = LearningConfig(learning_rate=0.01)
    net = ValueNetwork(learn, 'test_net.pkl')
    ai = ChessAI(net, cfg, learn)
    
    board = chess.Board()
    move, stats = ai.compute_best_move(board)
    print(f"Best: {move}")
    print(f"Stats: {json.dumps(stats, indent=2, default=str)}")
    
    net_stats = net.get_statistics()
    print(f"\nNetwork stats:")
    print(json.dumps(net_stats, indent=2))