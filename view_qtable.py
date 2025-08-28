import pickle
import pprint  # import pretty print module

# Load Q-table from file
with open("qtable.pkl", "rb") as f:
    data = pickle.load(f)

# Pretty-print the Q-table
#pprint.pprint(data["q_table"])

# Optional: print total episodes trained
print("Total episodes trained:", data.get("episodes", "Unknown"))
