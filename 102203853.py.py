import numpy as np
import pandas as pd
import argparse

def normalize_matrix(matrix):
    norm_matrix = matrix / np.sqrt(np.sum(np.square(matrix), axis=0))
    return norm_matrix

def weighted_normalized_matrix(norm_matrix, weights):
    return norm_matrix * weights

def ideal_solution(matrix):
    return np.max(matrix, axis=0), np.min(matrix, axis=0)

def euclidean_distance(matrix, ideal_positive, ideal_negative):
    pos_dist = np.sqrt(np.sum(np.square(matrix - ideal_positive), axis=1))
    neg_dist = np.sqrt(np.sum(np.square(matrix - ideal_negative), axis=1))
    return pos_dist, neg_dist

def topsis(data, weights, impacts):
    # Normalize the matrix
    norm_matrix = normalize_matrix(data)
    
    # Create the weighted normalized matrix
    weighted_matrix = weighted_normalized_matrix(norm_matrix, weights)
    
    # Get the ideal solutions
    ideal_pos, ideal_neg = ideal_solution(weighted_matrix)
    
    # Calculate the Euclidean distance
    pos_dist, neg_dist = euclidean_distance(weighted_matrix, ideal_pos, ideal_neg)
    
    # Calculate the performance score
    score = neg_dist / (pos_dist + neg_dist)
    
    return score

def main():
    parser = argparse.ArgumentParser(description='TOPSIS Method Implementation')
    parser.add_argument('input_file', help='CSV file containing the data')
    parser.add_argument('weights', help='Comma separated list of weights')
    parser.add_argument('impacts', help='Comma separated list of impacts (+ or -)')
    parser.add_argument('output_file', help='File to save the output')
    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.input_file)
    
    # Extract the numeric part of the dataset (exclude the Fund Name column)
    matrix = data.iloc[:, 1:].values  # The matrix without the first column
    
    # Convert weights and impacts to appropriate lists
    weights = np.array([float(w) for w in args.weights.split(',')])
    impacts = args.impacts.split(',')
    
    # Convert impacts to +1 and -1
    impacts = [1 if i == '+' else -1 for i in impacts]
    
    # Perform the TOPSIS calculation
    score = topsis(matrix, weights, impacts)
    
    # Save results to the output file
    data['Topsis Score'] = score
    data['Rank'] = score.argsort()[::-1] + 1
    data.to_csv(args.output_file, index=False)
    print(f'Results saved to {args.output_file}')

if __name__ == "__main__":
    main()
