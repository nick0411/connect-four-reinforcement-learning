import random
import pickle
import pygame
import sys
import numpy as np

ROWS, COLS = 6, 7

# ==================== ENVIRONMENT ====================
def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

def available_moves(board):
    return [c for c in range(COLS) if board[0][c] == 0]

def make_move(board, col, player):
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == 0:
            board[r][col] = player
            return True
    return False

def check_winner(board, player):
    # horizontal
    for r in range(ROWS):
        for c in range(COLS-3):
            if all(board[r][c+i] == player for i in range(4)):
                return True
    # vertical
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(board[r+i][c] == player for i in range(4)):
                return True
    # pos diag
    for r in range(ROWS-3):
        for c in range(COLS-3):
            if all(board[r+i][c+i] == player for i in range(4)):
                return True
    # neg diag
    for r in range(3, ROWS):
        for c in range(COLS-3):
            if all(board[r-i][c+i] == player for i in range(4)):
                return True
    return False

def board_full(board):
    return all(board[0][c] != 0 for c in range(COLS))

def encode_state(board):
    return tuple(board.reshape(-1))

# ==================== Q-LEARNING AGENT ====================
class QAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99995):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(COLS)
        return self.q_table[state]

    def choose_action(self, state, valid_moves, board=None, player=2):
        if board is not None:
            move = self.heuristic_action(board, player)
            if move is not None:
                return move

        if random.random() < self.epsilon or random.random() < 0.05:
            return random.choice(valid_moves)

        qs = self.get_qs(state)
        masked_qs = np.full(COLS, -1e9)
        masked_qs[valid_moves] = qs[valid_moves]
        return int(np.argmax(masked_qs))

    def heuristic_action(self, board, player):
        valid_moves = available_moves(board)
        opponent = 2 if player == 1 else 1

        # Immediate win
        for move in valid_moves:
            temp_board = board.copy()
            make_move(temp_board, move, player)
            if check_winner(temp_board, player):
                return move

        # Immediate block
        for move in valid_moves:
            temp_board = board.copy()
            make_move(temp_board, move, opponent)
            if check_winner(temp_board, opponent):
                return move

        return None

    def update(self, state, action, reward, next_state, done):
        qs = self.get_qs(state)
        if done:
            qs[action] += self.alpha * (reward - qs[action])
        else:
            next_qs = self.get_qs(next_state)
            qs[action] += self.alpha * (reward + self.gamma * np.max(next_qs) - qs[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# ==================== INTERMEDIATE REWARD ====================
def intermediate_reward(board, player):
    reward = 0
    opponent = 2 if player == 1 else 1
    directions = [(1,0),(0,1),(1,1),(1,-1)]

    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in directions:
                window = []
                for i in range(4):
                    nr, nc = r + dr*i, c + dc*i
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        window.append(board[nr][nc])
                    else:
                        break
                if len(window) == 4:
                    if window.count(player) == 3 and window.count(0) == 1:
                        reward += 0.5
                    elif window.count(player) == 2 and window.count(0) == 2:
                        reward += 0.2
                    if window.count(opponent) == 3 and window.count(0) == 1:
                        reward -= 0.4
    return reward

# ==================== SAVE / LOAD ====================
def save_agent(agent, total_episodes, history=None, filename="qtable.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({
            "q_table": agent.q_table,
            "episodes": total_episodes,
            "history": history if history is not None else []
        }, f)

def load_agent(agent, filename="qtable.pkl"):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            agent.q_table = data.get("q_table", {})
            total_episodes = data.get("episodes", 0)
            history = data.get("history", [])
        print(f"Loaded Q-table. Episodes: {total_episodes}, History length: {len(history)}")
        return total_episodes, history
    except FileNotFoundError:
        print("No saved Q-table found. Starting fresh.")
        return 0, []

# ==================== TRAIN ====================
def train(agent, episodes=100_000, total_episodes=0, save_file="qtable.pkl"):
    _, history = load_agent(agent, save_file)

    for ep in range(episodes):
        print(f"Episode {total_episodes + ep + 1}/{total_episodes + episodes}, Epsilon: {agent.epsilon:.4f}", end='\r')
        board = create_board()
        state = encode_state(board)
        done = False
        player = 1
        result = None

        while not done:
            valid_moves = available_moves(board)
            action = agent.choose_action(state, valid_moves, board=board, player=player)
            make_move(board, action, player)

            if check_winner(board, player):
                reward = 1 if player == 1 else -1
                agent.update(state, action, reward, None, True)
                result = "win" if player == 1 else "loss"
                done = True
            elif board_full(board):
                agent.update(state, action, 0.5, None, True)
                result = "draw"
                done = True
            else:
                next_state = encode_state(board)
                reward = intermediate_reward(board, player)
                agent.update(state, action, reward, next_state, False)
                state = next_state
                player = 2 if player == 1 else 1

        agent.decay_epsilon()
        history.append(result)

        if (ep + 1) % 10_000 == 0:
            agent.epsilon = 0.5
            save_agent(agent, total_episodes + ep + 1, history, save_file)

    total_episodes += episodes
    save_agent(agent, total_episodes, history, save_file)
    print(f"Training completed. Total episodes trained: {total_episodes}")
    return total_episodes, history

# ==================== PYGAME VISUALS ====================
CELL_SIZE = 100
WIDTH, HEIGHT = COLS * CELL_SIZE, (ROWS+1) * CELL_SIZE
RADIUS = CELL_SIZE//2 - 5

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

pygame.init()
FONT = pygame.font.SysFont("monospace", 40)

def draw_board(screen, board):
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c*CELL_SIZE, r*CELL_SIZE+CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.circle(screen, BLACK, (c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE+CELL_SIZE//2), RADIUS)

    for c in range(COLS):
        for r in range(ROWS):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE+CELL_SIZE//2), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE+CELL_SIZE//2), RADIUS)
    pygame.display.update()

# ==================== HUMAN VS AI ====================
def play_human_vs_ai(agent):
    load_agent(agent, "qtable.pkl")
    board = create_board()
    game_over = False
    turn = 0  # human first
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    draw_board(screen, board)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if turn == 0 and event.type == pygame.MOUSEBUTTONDOWN:
                x = event.pos[0]//CELL_SIZE
                if x in available_moves(board):
                    make_move(board, x, 1)
                    if check_winner(board, 1):
                        print("Human wins!")
                        game_over = True
                    turn = 1

        if turn == 1 and not game_over:
            state = encode_state(board)
            action = agent.choose_action(state, available_moves(board), board=board, player=2)
            make_move(board, action, 2)
            if check_winner(board, 2):
                print("AI wins!")
                game_over = True
            turn = 0

        draw_board(screen, board)

        if board_full(board) and not game_over:
            print("Draw!")
            game_over = True

        if game_over:
            pygame.time.wait(3000)

# ==================== AI VS AI ====================
def play_ai_vs_ai(agent, episodes=1, delay=300):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    for ep in range(episodes):
        board = create_board()
        game_over = False
        turn = 1
        draw_board(screen, board)
        pygame.time.wait(500)
        
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            
            state = encode_state(board)
            action = agent.choose_action(state, available_moves(board), board=board, player=turn)
            make_move(board, action, turn)
            
            if check_winner(board, turn):
                print(f"AI {turn} wins!")
                game_over = True
            elif board_full(board):
                print("Draw!")
                game_over = True
            
            draw_board(screen, board)
            pygame.time.wait(delay)
            turn = 2 if turn == 1 else 1
        
        pygame.time.wait(1000)

# ==================== MAIN ====================
if __name__ == "__main__":
    agent = QAgent()
    total_episodes, history = load_agent(agent, "qtable.pkl")

    total_episodes, history = train(agent, episodes=10000, total_episodes=total_episodes, save_file="qtable.pkl")

    #play_ai_vs_ai(agent, episodes=4, delay=300)
    #play_human_vs_ai(agent)
