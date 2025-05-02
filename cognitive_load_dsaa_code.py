import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import warnings

import textwrap  # <-- ADD THIS IMPORT AT THE TOP OF YOUR SCRIPT
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Set all seeds for reproducibility
SPLIT_SEED = 38
ALGO_SEED = 45
# np.random.seed(SEED)

def main(SPLIT_SEED=SPLIT_SEED, ALGO_SEED=ALGO_SEED):
    # Load data
    merged_df = pd.read_csv("all_Features_df.csv").drop(columns=['Unnamed: 0'])
    
    # Feature configuration
    modelling_feature = 'Demanding'
    y_feats = ['Effort', 'Satisfied', 'Demanding', 'Stress', 'Cognitive Load']
    excluded = set(y_feats) | {'Student ID', 'Exercise ID', 'Subcategory', 
                              'Solution', 'Prerequisite', 'Hint', 'Task', 
                              'Sample Input', 'Sample Output'}
    
    # Create ordered feature list
    X_feats = sorted(list(set(merged_df.columns) - excluded))
    
    # Split students
    unique_students = merged_df['Student ID'].unique()
    train_students, test_students = train_test_split(
        unique_students, test_size=0.2, random_state=SPLIT_SEED
    )
    
    # Create datasets
    train_df = merged_df[merged_df['Student ID'].isin(train_students)]
    test_df = merged_df[merged_df['Student ID'].isin(test_students)]

    # Check if there's similar rows in train and test sets
    train_test_check = train_df[train_df['Student ID'].isin(test_students)]
    if not train_test_check.empty:
        print("Warning: There are similar rows in train and test sets.")

    # Preprocessing setup
    categorical_features = ['Category', 'Difficulty']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Model definitions with enhanced reproducibility
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=ALGO_SEED, n_jobs=1
        ),
        'Support Vector Regressor': SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=ALGO_SEED),
        'XGBoost': XGBRegressor(random_state=ALGO_SEED, n_estimators=100, n_jobs=1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=ALGO_SEED),
        'KNN Regression': KNeighborsRegressor(n_neighbors=5),
        'Extra Trees': ExtraTreesRegressor(random_state=ALGO_SEED, n_estimators= 100, max_depth= None, min_samples_split=9, min_samples_leaf=2, max_features=0.9, bootstrap=False, criterion='absolute_error', ccp_alpha=0)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Handle missing values before fitting
        X_train = train_df[X_feats].fillna(0)
        y_train = train_df[modelling_feature].fillna(0)
        X_test = test_df[X_feats].fillna(0)
        y_test = test_df[modelling_feature].values
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # Modify your training loop:
        results[name] = {
            'RMSE': rmse, 
            'R2': r2, 
            'y_pred': y_pred,
            'pipeline': pipeline  # Store the entire pipeline
        }
        print(f"{name} - R2: {r2:.2f}, RMSE: {rmse:.2f}")

    # Plotting configuration
    plt.rcParams.update({
        'font.size': 20,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

    # Create combined prediction plot
    plt.figure(figsize=(15, 8))

    # Plot ground truth
    plt.plot(y_test, 
            label='Ground Truth', 
            color='green', 
            linestyle='-', 
            linewidth=3,
            marker='o', 
            markersize=8,
            alpha=0.7)

    # Define styling for each model
    model_styles = {
        'Random Forest': {'ls': '--', 'marker': 's', 'color': 'orange', 'markersize': 8},
        'XGBoost': {'ls': '-.', 'marker': 'D', 'color': 'blue', 'markersize': 7},
        'Extra Trees': {'ls': ':', 'marker': '^', 'color': 'red', 'markersize': 9}
    }

    # Plot predictions for each model
    for name, style in model_styles.items():
        if name in results:
            y_pred = results[name]['y_pred']
            rmse = results[name]['RMSE']
            plt.plot(y_pred,
                    linestyle=style['ls'],
                    marker=style['marker'],
                    color=style['color'],
                    markersize=style['markersize'],
                    linewidth=2.5,
                    alpha=0.8,
                    label=f"{name} (RMSE: {rmse:.2f})")

    plt.xlabel('Sample Index', weight='bold')
    plt.ylabel('Demanding Score', weight='bold')
    plt.title('Model Predictions Comparison', weight='bold', pad=16)

    plt.legend(
        fontsize=16,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        ncol=3,
        frameon=True,
        framealpha=0.2,
        facecolor='silver'
    )

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs("figures/ml_modelling/", exist_ok=True)
    plt.savefig("figures/ml_modelling/combined_predictions.eps", 
               dpi=300, 
               format='eps',
               bbox_inches='tight')
    plt.show()


    # Feature Importance Plotting
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'sans-serif',
        'axes.titlesize': 9,
        'axes.labelsize': 8
    })

    # Get the trained pipeline for Extra Trees
    et_result = results['Extra Trees']
    et_pipeline = et_result['pipeline']

    # Get proper feature names from preprocessing
    preprocessor = et_pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()

    # Get feature importance from ACTUAL trained model
    trained_model = et_pipeline.named_steps['model']
    feature_importance = trained_model.feature_importances_

    # Process features
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices[:20]] 
    sorted_importance = feature_importance[sorted_indices[:20]]

    # Create compact figure
    fig, ax = plt.subplots(figsize=(5, 5)) 

    # Horizontal bar plot (rest of your plotting code remains the same)
    ax.barh(
        np.arange(20)[::-1],
        sorted_importance,
        height=0.6,
        color='#1f77b4',
        edgecolor='w'
    )

    # Feature labels with smart wrapping
    max_chars_per_line = 40
    wrapped_labels = [
        '\n'.join(textwrap.wrap(f, max_chars_per_line, break_long_words=False))
        for f in sorted_features
    ]

    ax.set_yticks(np.arange(20)[::-1])  # Explicit 20 instead of len(sorted_features) ⬅️
    ax.set_yticklabels(wrapped_labels)
    ax.set_xlabel('Importance Score', labelpad=4, fontsize = 9, fontweight='bold')

    # Adjust grid and spines
    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(
        "figures/ml_modelling/feature_importance.eps",
        dpi=300,
        format='eps',
        bbox_inches='tight'
    )
    plt.show()


if __name__ == "__main__":
    main()
