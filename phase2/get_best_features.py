import pandas as pd
import numpy as np
import json
from scipy.stats import zscore
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def load_model_results(file_path):
    """ Load model results from a JSON file and return the F1 score. """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['f1']

def load_and_process_importances(file_path, f1_score, subset_tag, codebook):
    """ Load feature importances, compute z-scores, weigh by F1 score, and rename features based on codebook. """
    df = pd.read_csv(file_path)
    sorted_codebook = sorted(codebook['var_name'].tolist(), reverse=True, key=len)
    #sorted_codebook = codebook['var_name'].sort_values(key=len, ascending=False).tolist()

    def find_longest_prefix(feature):
        for var_name in sorted_codebook:
            if feature.startswith(var_name):
                return var_name
        assert(False)
        return feature
    
    df['feature'] = df['feature'].apply(find_longest_prefix)
    df['z_score'] = zscore(df['importance'])
    df['weighted_z'] = df['z_score'] * (f1_score ** 2)
    df['subset'] = subset_tag  # Tag each feature with its subset identifier
    return df[['feature', 'weighted_z', 'subset']]

def process_files(result_dir, importance_dir):
    all_features = pd.DataFrame()
    result_files = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
    importance_files = [f for f in listdir(importance_dir) if isfile(join(importance_dir, f))]
    codebook = pd.read_csv('../PreFer_codebook.csv')
    codebook['var_name'] = codebook['var_name'].astype(str)
    for result_file, importance_file in tqdm(zip(sorted(result_files), sorted(importance_files))):
        f1_score = load_model_results(join(result_dir, result_file))
        subset_tag = result_file.split('.')[0]  # Use the file name prefix as the subset identifier
        feature_data = load_and_process_importances(join(importance_dir, importance_file), f1_score, subset_tag, codebook)
        all_features = pd.concat([all_features, feature_data])
    
    return all_features

def main():
    result_dir = 'result/'
    importance_dir = 'importance/'
    all_features = process_files(result_dir, importance_dir)
    final_scores = all_features.groupby('feature').sum().reset_index()
    print(len(final_scores))

    top_features = final_scores.nlargest(200, 'weighted_z')

    # Remove duplicates and keep only the first occurrence for subset data
    all_features_unique = all_features.drop_duplicates(subset='feature', keep='first')

    # Merge the subset information into top_features for the top 1000 features
    top_features = top_features.merge(all_features_unique[['feature', 'subset']], on='feature', how='left')

    # Calculate the percentage from each subset in the top features
    subset_counts = top_features['subset'].value_counts(normalize=True) * 100

    # Export the top features with subset information to a CSV file
    top_features.to_csv('../data/top_200.csv', index=False)
    
    print("Percentage of selected features from each subset:")
    print(subset_counts)

# Ensure to call the main function if this script is executed as the main program
if __name__ == "__main__":
    main()

