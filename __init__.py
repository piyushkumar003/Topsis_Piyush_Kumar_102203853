import pandas as pd
import numpy as np
import sys

def validate_inputs(input_file, weights, impacts):
    # Check if the file exists
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Error: Input file not found.")
        sys.exit(1)
    
    # Validate file format and structure
    if data.shape[1] < 3:
        print("Error: Input file must have at least 3 columns.")
        sys.exit(1)
    
    if not all(data.iloc[:, 1:].applymap(np.isreal).all(axis=1)):
        print("Error: All columns except the first must contain numeric values.")
        sys.exit(1)
    
    # Validate weights and impacts
    weights = weights.split(",")
    impacts = impacts.split(",")
    if len(weights) != (data.shape[1] - 1) or len(impacts) != (data.shape[1] - 1):
        print("Error: Number of weights and impacts must match the number of criteria (excluding the first column).")
        sys.exit(1)
    
    if not all(w.isdigit() for w in weights):
        print("Error: Weights must be numeric values.")
        sys.exit(1)
    
    if not all(i in ['+', '-'] for i in impacts):
        print("Error: Impacts must be either '+' or '-'.")
        sys.exit(1)
    
    return data, list(map(float, weights)), impacts

def topsis(input_file, weights, impacts, result_file):
    # Validate inputs and load data
    data, weights, impacts = validate_inputs(input_file, weights, impacts)
    
    # Normalize the dataset (excluding the first column)
    matrix = data.iloc[:, 1:].values
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    
    # Apply weights
    weighted_matrix = norm_matrix * weights
    
    # Determine ideal best and worst values
    ideal_best = []
    ideal_worst = []
    for i, impact in enumerate(impacts):
        if impact == '+':
            ideal_best.append(np.max(weighted_matrix[:, i]))
            ideal_worst.append(np.min(weighted_matrix[:, i]))
        elif impact == '-':
            ideal_best.append(np.min(weighted_matrix[:, i]))
            ideal_worst.append(np.max(weighted_matrix[:, i]))
    
    # Calculate Euclidean distances
    dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    # Calculate Topsis score
    scores = dist_worst / (dist_best + dist_worst)
    
    # Add Topsis score and rank to the result
    data['Topsis Score'] = scores
    data['Rank'] = pd.Series(scores).rank(ascending=False)
    
    # Save to result file
    data.to_csv(result_file, index=False)
    print(f"Result file '{result_file}' has been created successfully.")

if __name__ == "__main__":
    # Command-line arguments: python <script.py> <inputFile> <weights> <impacts> <resultFile>
    if len(sys.argv) != 5:
        print("Usage: python <script.py> <inputFile> <weights> <impacts> <resultFile>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    result_file = sys.argv[4]
    
    topsis(input_file, weights, impacts, result_file)
