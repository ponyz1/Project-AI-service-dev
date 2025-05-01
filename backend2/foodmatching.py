#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Food Matching Prediction System

This module provides functions to train models that predict how well
different foods pair together based on their flavor profiles.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("XGBoost not installed. XGBoost model will not be available.")
import matplotlib.pyplot as plt
import joblib
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(food_data_path="food_data.csv", match_data_path="match_rating.csv"):
    """
    Load food data and match rating data from CSV files
    
    Parameters:
    -----------
    food_data_path : str
        Path to CSV file containing food flavor profiles
    match_data_path : str
        Path to CSV file containing food match ratings
        
    Returns:
    --------
    tuple
        (food_data, match_data) - Pandas DataFrames
    """
    food_data = pd.read_csv(food_data_path)
    match_data = pd.read_csv(match_data_path)
    
    print(f"Loaded {len(food_data)} foods and {len(match_data)} ratings")
    
    return food_data, match_data


def prepare_features(food_data, match_data):
    """
    Create feature vectors for each food pair
    
    Parameters:
    -----------
    food_data : pandas.DataFrame
        DataFrame containing food flavor profiles
    match_data : pandas.DataFrame
        DataFrame containing food match ratings
        
    Returns:
    --------
    tuple
        (X, y, feature_names) - Feature matrix, target values, and feature names
    """
    feature_vectors = []
    ratings = []
    flavor_cols = ['spicy', 'sweet', 'sour', 'salty', 'texture']
    
    for _, row in match_data.iterrows():
        food1 = row['food1']
        food2 = row['food2']
        rating = row['rating']
        
        # Get flavor profiles for each food
        food1_profile = food_data[food_data['name'] == food1]
        food2_profile = food_data[food_data['name'] == food2]
        
        # Skip if either food is not in the food_data
        if len(food1_profile) == 0 or len(food2_profile) == 0:
            continue
            
        food1_profile = food1_profile.iloc[0]
        food2_profile = food2_profile.iloc[0]
        
        # Extract flavor features
        f1 = food1_profile[flavor_cols].values
        f2 = food2_profile[flavor_cols].values
        
        # Difference and interaction features
        diff = np.abs(f1 - f2)  # Absolute differences
        product = f1 * f2       # Interactions
        
        # Food type information (encoded as binary: same_type or different_type)
        same_type = 1 if food1_profile['type'] == food2_profile['type'] else 0
        
        # Combine all features
        features = np.concatenate([
            f1,                  # Food 1 flavors
            f2,                  # Food 2 flavors
            diff,                # Differences
            product,             # Interactions
            [same_type]          # Type comparison
        ])
        
        feature_vectors.append(features)
        ratings.append(rating)
    
    # Convert to numpy arrays
    X = np.array(feature_vectors)
    y = np.array(ratings)
    
    # Store feature names for later interpretation
    feature_names = (
        [f'food1_{col}' for col in flavor_cols] +
        [f'food2_{col}' for col in flavor_cols] +
        [f'diff_{col}' for col in flavor_cols] +
        [f'product_{col}' for col in flavor_cols] +
        ['same_type']
    )
    
    print(f"X: {X.shape[0]} samples and {X.shape[1]} features")
    
    return X, y, feature_names


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : trained model object
        The trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(8, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()
        plt.close()

        print("Feature importance plot saved as 'feature_importance.png'")

        # Print top 10 important features
        print("\nTop 10 important features:")
        for i in range(min(10, len(indices))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


def train_model(X, y, model_type='gradient_boosting', scaler=None):
    """
    Train a model to predict food matching ratings

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target values (ratings)
    model_type : str
        Type of model to train ('random_forest', 'gradient_boosting', 'neural_network', 'xgboost')
    scaler : StandardScaler, optional
        Fitted scaler to use. If None, a new scaler will be created.

    Returns:
    --------
    tuple
        (model, scaler, metrics) - The trained model, fitted scaler, and evaluation metrics
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize or use provided scaler
    if scaler is None:
        scaler = StandardScaler()
        
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model based on type
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif model_type == 'neural_network':
        model = MLPRegressor(random_state=42, max_iter=2000)
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        }
    elif model_type == 'xgboost':
        if xgb is None:
            raise ValueError("XGBoost is not installed. Please install xgboost package or use a different model type.")
        model = xgb.XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use grid search to find best hyperparameters
    print(f"Training {model_type} model with grid search...")
    try:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
    except Exception as e:
        print(f"Grid search failed with error: {str(e)}")
        print("Falling back to default parameters...")
        # Fall back to training with default parameters if grid search fails
        model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy within 0.5 and 1.0 points
    accuracy_05 = np.mean(np.abs(y_test - y_pred) <= 0.5)
    accuracy_10 = np.mean(np.abs(y_test - y_pred) <= 1.0)
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'accuracy_within_0.5': accuracy_05,
        'accuracy_within_1.0': accuracy_10,
        'best_params': grid_search.best_params_
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Accuracy (within 0.5): {accuracy_05:.2%}")
    print(f"Accuracy (within 1.0): {accuracy_10:.2%}")
    
    return model, scaler, metrics


def evaluate_model(model, X, y, scaler):
    """
    Evaluate a trained model on given data
    
    Parameters:
    -----------
    model : trained model object
        The trained model to evaluate
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target values (ratings)
    scaler : StandardScaler
        Fitted scaler to use
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    if model is None:
        raise ValueError("Model not trained yet")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate accuracy within 0.5 and 1.0 points
    accuracy_05 = np.mean(np.abs(y - y_pred) <= 0.5)
    accuracy_10 = np.mean(np.abs(y - y_pred) <= 1.0)
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'accuracy_within_0.5': accuracy_05,
        'accuracy_within_1.0': accuracy_10
    }
    
    print(f"Evaluation metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy (within 0.5): {accuracy_05:.2%}")
    print(f"Accuracy (within 1.0): {accuracy_10:.2%}")
    
    return metrics


def predict(model, food1_features, food2_features, scaler, food1_type=None, food2_type=None):
    """
    Predict the matching rating for two foods
    
    Parameters:
    -----------
    model : trained model object
        The trained model to use for prediction
    food1_features : array-like
        Flavor profile of food 1 [spicy, sweet, sour, salty, texture]
    food2_features : array-like
        Flavor profile of food 2 [spicy, sweet, sour, salty, texture]
    scaler : StandardScaler
        Fitted scaler to use
    food1_type : str, optional
        Type of food 1
    food2_type : str, optional
        Type of food 2
        
    Returns:
    --------
    float
        Predicted matching rating
    """
    if model is None:
        raise ValueError("Model not trained yet. Call train_model first.")
    
    # Convert inputs to numpy arrays
    food1_features = np.array(food1_features)
    food2_features = np.array(food2_features)
    
    # Calculate difference and product features
    diff = np.abs(food1_features - food2_features)
    product = food1_features * food2_features
    
    # Set same_type feature
    same_type = 1 if food1_type == food2_type else 0
    
    # Combine features
    features = np.concatenate([
        food1_features,
        food2_features,
        diff,
        product,
        [same_type]
    ]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    # Ensure prediction is within valid range
    prediction = max(1, min(5, prediction))
    
    return prediction


def predict_from_names(model, food1_name, food2_name, food_data, scaler):
    """
    Predict matching rating given food names and food data
    
    Parameters:
    -----------
    model : trained model object
        The trained model to use for prediction
    food1_name : str
        Name of food 1
    food2_name : str
        Name of food 2
    food_data : pandas.DataFrame
        DataFrame containing food flavor profiles
    scaler : StandardScaler
        Fitted scaler to use
        
    Returns:
    --------
    float
        Predicted matching rating
    """
    # Get food profiles
    food1_profile = food_data[food_data['name'] == food1_name]
    food2_profile = food_data[food_data['name'] == food2_name]
    
    if len(food1_profile) == 0:
        raise ValueError(f"Food '{food1_name}' not found in food data")
    if len(food2_profile) == 0:
        raise ValueError(f"Food '{food2_name}' not found in food data")
    
    food1_profile = food1_profile.iloc[0]
    food2_profile = food2_profile.iloc[0]
    
    # Extract flavor features
    flavor_cols = ['spicy', 'sweet', 'sour', 'salty', 'texture']
    food1_features = food1_profile[flavor_cols].values
    food2_features = food2_profile[flavor_cols].values
    
    # Get food types
    food1_type = food1_profile['type']
    food2_type = food2_profile['type']
    
    # Predict
    return predict(model, food1_features, food2_features, scaler, food1_type, food2_type)


def save_model(model, scaler, food_data, output_path='food_matching_model.pkl', save_format='pickle'):
    """
    Save trained model and related data
    
    Parameters:
    -----------
    model : trained model object
        The trained model to save
    scaler : StandardScaler
        Fitted scaler to use
    food_data : pandas.DataFrame
        DataFrame containing food flavor profiles
    output_path : str
        Path where the model will be saved
    save_format : str
        Format to use for saving ('pickle' or 'joblib')
        
    Returns:
    --------
    str
        Path to saved model file
    """
    # Create a copy of the original food data and set index
    original_food_data = food_data.copy()
    original_food_data.set_index('name', inplace=True)
    
    # Create feature dataframe for easy lookup
    flavor_cols = ['spicy', 'sweet', 'sour', 'salty', 'texture']
    food_feature_df = food_data[flavor_cols].copy()
    food_feature_df.index = food_data['name']
    
    # Create feature names for reference
    feature_names = (
        [f'food1_{col}' for col in flavor_cols] +
        [f'food2_{col}' for col in flavor_cols] +
        [f'diff_{col}' for col in flavor_cols] +
        [f'product_{col}' for col in flavor_cols] +
        ['same_type']
    )
    
    # Create model data dictionary
    model_data = {
        'model': model,
        'scaler': scaler,
        'food_data': food_feature_df,
        'original_food_data': original_food_data,
        'feature_names': feature_names
    }
    
    if save_format == 'joblib':
        joblib.dump(model_data, output_path)
        print(f"Model saved to {output_path} using joblib")
    else:
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=4)  # Use protocol 4 for better compatibility
        print(f"Model saved to {output_path} using pickle")
    
    # Verify the saved model can be loaded
    try:
        if save_format == 'joblib':
            loaded_data = joblib.load(output_path)
        else:
            with open(output_path, 'rb') as f:
                loaded_data = pickle.load(f)
        print(f"✓ Successfully verified model loading from {output_path}")
        return output_path
    except Exception as e:
        print(f"⚠ Error verifying model: {str(e)}")
        return None


def load_model(model_path):
    """
    Load a saved model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
        
    Returns:
    --------
    dict
        Dictionary containing model and related data
    """
    try:
        # Try to load with joblib first
        try:
            model_data = joblib.load(model_path)
            print(f"Model loaded from {model_path} using joblib")
        except:
            # If joblib fails, try pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"Model loaded from {model_path} using pickle")
        
        return model_data
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def run_training_pipeline(model_types=None):
    """
    Run the complete training pipeline and save the best model
    
    Parameters:
    -----------
    model_types : list, optional
        List of model types to train. If None, will try all available models.
        
    Returns:
    --------
    dict
        Dictionary containing the best model and related data
    """
    # Load data
    food_data, match_data = load_data()
    
    # Prepare features
    X, y, feature_names = prepare_features(food_data, match_data)
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Default model types to train
    if model_types is None:
        model_types = ['gradient_boosting', 'random_forest', 'neural_network']
        if xgb is not None:
            model_types.append('xgboost')
    
    # Dictionary to store trained models
    models = {}
    
    # Train specified models
    for model_type in model_types:
        try:
            print(f"\n=== Training {model_type.upper()} model ===")
            model, model_scaler, metrics = train_model(X, y, model_type=model_type, scaler=scaler)
            models[model_type] = (model, model_scaler, metrics)
        except Exception as e:
            print(f"Error training {model_type} model: {str(e)}")
    
    # Check if any models were successfully trained
    if not models:
        raise ValueError("No models were successfully trained. Please check error messages.")
    
    # Choose best model based on MSE
    best_model_name = min(models, key=lambda k: models[k][2]['mse'])
    best_model, best_scaler, best_metrics = models[best_model_name]
    
    print(f"\n=== Best model: {best_model_name.upper()} ===")
    print(f"MSE: {best_metrics['mse']:.4f}")
    print(f"R²: {best_metrics['r2']:.4f}")
    
    # Plot feature importance for best model
    if best_model_name in ['xgboost', 'random_forest', 'gradient_boosting']:
        plot_feature_importance(best_model, feature_names)
    
    # Save best model
    save_model(best_model, best_scaler, food_data, output_path='best_food_matching_model.pkl')
    
    return {
        'model': best_model,
        'scaler': best_scaler,
        'food_data': food_data,
        'feature_names': feature_names,
        'metrics': best_metrics
    }


def single_model_training(model_type='gradient_boosting'):
    """
    Train a single model of specified type
    
    Parameters:
    -----------
    model_type : str
        Type of model to train
        
    Returns:
    --------
    dict
        Dictionary containing the model and related data
    """
    # Load data
    food_data, match_data = load_data()
    
    # Prepare features
    X, y, feature_names = prepare_features(food_data, match_data)
    
    # Train model
    print(f"\n=== Training {model_type.upper()} model ===")
    model, scaler, metrics = train_model(X, y, model_type=model_type)
    
    # Save model
    save_model(model, scaler, food_data, output_path=f'model.pkl')
    
    return {
        'model': model,
        'scaler': scaler,
        'food_data': food_data,
        'feature_names': feature_names,
        'metrics': metrics
    }


if __name__ == '__main__':
    try:
        # Run training pipeline when script is executed directly
        model_data = single_model_training('gradient_boosting')
        # Generate some food pairs for demonstration
        food_data = model_data['food_data']
        model = model_data['model']
        scaler = model_data['scaler']
        
        print("\n=== Testing model with example food pairs ===")
        food_pairs = [
            # Try different combinations: same type, different types
            ("น้ำมะนาว", "หมูแดดเดียว"),
            (food_data.iloc[2]['name'], food_data.iloc[22]['name']),
            (food_data.iloc[3]['name'], food_data.iloc[10]['name']),
        ]
        
        for food1_name, food2_name in food_pairs:
            try:
                prediction = predict_from_names(model, food1_name, food2_name, food_data, scaler)
                food1_type = food_data[food_data['name'] == food1_name].iloc[0]['type']
                food2_type = food_data[food_data['name'] == food2_name].iloc[0]['type']
                
                print(f"{food1_name} ({food1_type}) + {food2_name} ({food2_type}): {prediction:.2f}/5")
            except Exception as e:
                print(f"Error predicting {food1_name} + {food2_name}: {e}")
                
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        print("\nFallback to simple model training...")
        # Fallback to simple training without grid search
        food_data, match_data = load_data()
        X, y, feature_names = prepare_features(food_data, match_data)
        
        print("Training a simple Gradient Boosting model...")
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        save_model(model, scaler, food_data, output_path='fallback_model.pkl')
        print("Fallback model saved as 'fallback_model.pkl'")