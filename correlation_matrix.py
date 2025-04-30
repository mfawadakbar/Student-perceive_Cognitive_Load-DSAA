import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings("ignore")

def main(merged_df, X_feats, modelling_feature='Demanding'):
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

    # Select relevant features from the dataframe
    df_selected = merged_df[X_feats + ['Demanding', 'Effort', 'Stress', 'Satisfied']].copy()

    # Identify categorical features
    categorical_features = ['Category', 'Difficulty']  # Add any other categorical features if needed

    # Convert categorical variables using one-hot encoding
    df_encoded = pd.get_dummies(df_selected, columns=categorical_features, drop_first=True)

    # Compute correlation matrix
    correlation_matrix = df_encoded.corr()

    # Get the absolute correlations with the 'demanding' variable
    correlation_with_demanding = correlation_matrix[modelling_feature].abs()

    # Select top 20 features with the highest correlation
    top_20_features = correlation_with_demanding.nlargest(20).index

    # Extract the correlation matrix for these top features
    top_corr_matrix = df_encoded[top_20_features].corr()

    # Plot the heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(top_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    # plt.title(f"Correlation Matrix of Top 20 Features Most Correlated with {modelling_feature}")
    # plt.savefig("figures/correlation/correlation_matrix.eps", format="eps", dpi = 600)
    plt.show()

if __name__ == "__main__":
    # Load data once
    merged_df = pd.read_csv("all_Features_df.csv").drop(columns=['Unnamed: 0'])

    # Define features to be used for modelling
    modelling_feature = 'Demanding'
    y_feats = ['Effort', 'Satisfied', 'Demanding', 'Stress', 'Cognitive Load']
    excluded = set(y_feats) | {'Student ID', 'Exercise ID', 'Subcategory', 
                                'Solution', 'Prerequisite', 'Hint', 'Task', 
                                'Sample Input', 'Sample Output'}
    
    # Create ordered feature list
    X_feats = sorted(list(set(merged_df.columns) - excluded))

    print(f"Features used for modelling: {X_feats}")

    main(merged_df, X_feats, modelling_feature)