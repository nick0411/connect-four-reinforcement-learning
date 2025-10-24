import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================
# Load Training History
# ============================
def load_history(filename="qtable.pkl.gz"):
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return []

    try:
        # Handle gzip-compressed pickle file
        with gzip.open(filename, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print("❌ Failed to load:", e)
        return []

    # Extract history list
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    elif isinstance(data, list):
        return data
    else:
        print("⚠️ No valid history found in file.")
        return []

# ============================
# Plot Reward Curve (numeric)
# ============================
def plot_reward_curve(history, window=1000):
    plt.figure(figsize=(10, 5))
    plt.plot(history, alpha=0.4, label="Raw Rewards")

    if len(history) > window:
        smoothed = np.convolve(history, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(history)), smoothed,
                 label=f"Smoothed (window={window})", color='red')

    plt.title("Training Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================
# Plot Win Rate (for 'win/loss/draw')
# ============================
def plot_win_rate(history, window=1000):
    results = {"win": 1, "loss": 0, "draw": 0.5}
    numeric = [results.get(h, None) for h in history if h in results]

    if not numeric:
        print("⚠️ No win/loss/draw results found — skipping win rate plot.")
        return

    df = pd.DataFrame(numeric, columns=["result"])
    df["rolling_winrate"] = df["result"].rolling(window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["rolling_winrate"], label=f"Rolling Win Rate (window={window})", color="green")
    plt.title("AI Win Rate Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================
# Plot Epsilon (if available)
# ============================
def plot_epsilon(epsilons):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons, label="Epsilon Decay", color="purple")
    plt.title("Exploration Rate (Epsilon) Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================
# Main Entry Point
# ============================
if __name__ == "__main__":
    history = load_history("qtable.pkl.gz")
    print(f"✅ Loaded {len(history)} entries from training history.")

    if not history:
        exit()

    # Detect what kind of data we have
    if isinstance(history[0], (int, float)):
        print("📊 Detected numeric reward history.")
        plot_reward_curve(history)

    elif isinstance(history[0], str):
        print("🏆 Detected win/loss/draw history.")
        plot_win_rate(history, window=500)

    elif isinstance(history[0], dict):
        print("📁 Detected structured training data.")
        df = pd.DataFrame(history)
        print("Available columns:", list(df.columns))

        if "reward" in df.columns:
            plot_reward_curve(df["reward"])
        if "win" in df.columns:
            df["win_numeric"] = df["win"].astype(int)
            df["rolling_winrate"] = df["win_numeric"].rolling(1000).mean()
            plt.figure(figsize=(10, 5))
            plt.plot(df["rolling_winrate"], color='green', label="Rolling Win Rate")
            plt.title("Win Rate (1000-Episode Rolling Average)")
            plt.xlabel("Episode")
            plt.ylabel("Win Rate")
            plt.legend()
            plt.grid(True)
            plt.show()
        if "epsilon" in df.columns:
            plot_epsilon(df["epsilon"])

    else:
        print("⚠️ Unrecognized data format in training file.")
