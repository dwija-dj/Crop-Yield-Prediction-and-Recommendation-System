import pickle

# Load the existing feature_info.pkl
with open('feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)

# Update model names in the performance_metrics dictionary with a flat layout
feature_info['performance_metrics'] = {
    'random_forest_model': {
        'R2 Score': 0.9788575810068678,
        'RMSE': 130.15432738825623,
        'MAE': 9.433323718896673,
        'MSE': 16940.14893788939
    },
    'linear_regression_model': {
        'R2 Score': 0.031154367724477106,
        'RMSE': 881.0662537637671,
        'MAE': 209.84310956535327,
        'MSE': 776277.7435213189
    },
    'decision_tree_model': {
        'R2 Score': 0.9413196805585629,
        'RMSE': 216.834063795396,
        'MAE': 12.589355107374809,
        'MSE': 47017.01122202586
    },
    'gradient_boosting_model': {
        'R2 Score': 0.9432591020582806,
        'RMSE': 213.2207058462945,
        'MAE': 25.405533804801117,
        'MSE': 45463.069401592045
    },
    'support_vector_model': {
        'R2 Score': -0.007536444307598034,
        'RMSE': 898.4867080414081,
        'MAE': 78.5474118381256,
        'MSE': 807278.3645270865
    },
    'k-nearest_neighbors_model': {
        'R2 Score': 0.9479897026453362,
        'RMSE': 204.13898115324523,
        'MAE': 14.57240070446191,
        'MSE': 41672.72362628501
    }
}

# Save the updated feature_info back to feature_info.pkl
with open('feature_info.pkl', 'wb') as f:
    pickle.dump(feature_info, f)

print("feature_info.pkl has been updated successfully.")