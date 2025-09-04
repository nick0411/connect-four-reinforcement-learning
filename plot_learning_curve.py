import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_history(filename="qtable.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data.get("history", [])

def plot_learning_curve(history, window=1000):
    results = {"win": 1, "loss": 0, "draw": 0.5}
    numeric = [results[h] for h in history]

    if len(numeric) < window:
        window = max(1, len(numeric)//10)

    rolling_avg = np.convolve(numeric, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10,5))
    plt.plot(rolling_avg, label=f"Rolling Average (window={window})")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.title("Connect Four AI Learning Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    history = load_history("qtable.pkl")
    print(f"Loaded history length: {len(history)}")
    if history:
        plot_learning_curve(history)
    else:
        print("No history found in qtable.pkl")
