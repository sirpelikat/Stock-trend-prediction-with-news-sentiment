import pandas as pd
import numpy as np

# 1. Load the dataset
# We assume the CSV has headers like "Trend" (Hidden State) and "Observation" (Visible State)
try:
    df = pd.read_csv('stock_trend.csv')
    print("Dataset Loaded Successfully")
    print(df.head())
    print("-" * 30)
except FileNotFoundError:
    print("Error: 'stock_trend.csv' not found. Please check the file path.")
    exit()

# === CONFIGURATION ===
# PLEASE CHECK YOUR CSV COLUMNS AND UPDATE THESE NAMES IF NEEDED
# Based on typical assignments, the 'Hidden State' is usually "Trend" or "State"
# The 'Observation' is usually "Volume", "Activity", or "PriceChange"
state_col = df.columns[0]      # Assuming 1st column is the Hidden State (e.g., Up/Down)
obs_col = df.columns[1]        # Assuming 2nd column is the Observation (e.g., L/M/H)

print(f"Using '{state_col}' as Hidden State and '{obs_col}' as Observation.\n")
# 2. Calculate Start Probabilities (pi)
print("### 1. Start Probabilities (pi) ###")
start_counts = df[state_col].value_counts(normalize=True).sort_index()
print(start_counts)
print("-" * 30)

# 3. Calculate Transition Matrix (A)
# P(State_t | State_t-1)
print("### 2. Transition Matrix (A) ###")
# Create a new column for the 'Next State' to compare t and t+1
df['Next_State'] = df[state_col].shift(-1)

# Drop the last row as it has no 'Next State'
transition_data = df.dropna()

# Create cross-tabulation (Count matrix)
trans_counts = pd.crosstab(transition_data[state_col], transition_data['Next_State'])

# Normalize to get probabilities (Row sums = 1)
trans_probs = trans_counts.div(trans_counts.sum(axis=1), axis=0)
print(trans_probs)
print("\nInterpretation: Row is 'From', Column is 'To'. E.g., Row 1 Col 2 is P(State2 | State1)")
print("-" * 30)

# 4. Calculate Emission Matrix (B)
# P(Observation | State)
print("### 3. Emission Matrix (B) ###")
# Create cross-tabulation
emission_counts = pd.crosstab(df[state_col], df[obs_col])

# Normalize to get probabilities (Row sums = 1)
emission_probs = emission_counts.div(emission_counts.sum(axis=1), axis=0)
print(emission_probs)
print("\nInterpretation: Row is 'State', Column is 'Observation'. E.g., P(Obs | State)")
print("-" * 30)