#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import sys
import json
import io
import argparse

# Ensure UTF-8 encoding for stdout to handle Thai characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_model(model_path):
    """
    Load the model and associated data from pickle or joblib file
    """
    try:
        # Try to load with joblib first (recommended for scikit-learn models)
        model_data = joblib.load(model_path)
        print(f"Model loaded successfully with joblib from {model_path}", file=sys.stderr)
    except:
        try:
            # Fall back to pickle if joblib fails
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"Model loaded successfully with pickle from {model_path}", file=sys.stderr)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    return model_data

def prepare_features(input_food, target_food, food_data, original_food_data=None):
    """
    Prepare feature vector for model prediction based on two foods
    """
    # Get flavor profiles for both foods
    if original_food_data is not None and isinstance(food_data, pd.DataFrame):
        # If we have separate dataframes for features and metadata
        flavor_cols = ['spicy', 'sweet', 'sour', 'salty', 'texture']
        
        # Ensure input_food is in food_data
        if input_food not in food_data.index:
            raise ValueError(f"Food '{input_food}' not found in dataset")
            
        # Ensure target_food is in food_data
        if target_food not in food_data.index:
            raise ValueError(f"Food '{target_food}' not found in dataset")
        
        # Extract feature vectors
        food1_features = food_data.loc[input_food].values
        food2_features = food_data.loc[target_food].values
        
        # Get food types from original data
        if original_food_data is not None:
            food1_type = original_food_data.loc[input_food, 'type']
            food2_type = original_food_data.loc[target_food, 'type']
        else:
            # If no original data, just use empty strings
            food1_type = ""
            food2_type = ""
        
        # Create difference and interaction features
        diff = np.abs(food1_features - food2_features)
        product = food1_features * food2_features
        
        # Create same_type feature
        same_type = 1 if food1_type == food2_type else 0
        
        # Combine all features
        features = np.concatenate([
            food1_features,
            food2_features,
            diff,
            product,
            [same_type]
        ])
        
        return features
    else:
        # For simpler datasets, just concatenate the feature vectors
        food1_features = food_data.loc[input_food].values
        food2_features = food_data.loc[target_food].values
        return np.concatenate([food1_features, food2_features])

def recommend_by_type(model_data, input_food, food_type, top_n=3):
    """
    Get food recommendations of a specific type to match with input_food
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing the model and data
    input_food : str
        Name of the input food
    food_type : str
        Type of food to recommend
    top_n : int
        Number of recommendations to return
    
    Returns:
    --------
    dict
        Dictionary with recommendations
    """
    # Extract model and data from model_data
    model = model_data['model']
    
    # Determine which dataframes we have (might vary based on how model was saved)
    if 'food_data' in model_data and 'original_food_data' in model_data:
        food_df = model_data['food_data']
        original_df = model_data['original_food_data']
    elif 'food_data' in model_data:
        food_df = model_data['food_data']
        original_df = model_data['food_data']
    else:
        return {"error": "Invalid model data format. Missing food_data."}
    
    # Check if scaler was saved
    scaler = model_data.get('scaler', None)
    
    # Validate input food exists
    if input_food not in original_df.index:
        return {"error": f"ไม่พบเมนู '{input_food}'"}
    
    # Validate food type exists
    available_types = original_df['type'].unique()
    if food_type not in available_types:
        return {"error": f"ไม่พบประเภท '{food_type}'"}
    
    # Get foods of the target type (excluding the input food if it's of the same type)
    target_foods = original_df[(original_df['type'] == food_type) & (original_df.index != input_food)]
    
    if len(target_foods) == 0:
        return {"error": f"ไม่มีเมนูประเภท '{food_type}'"}
    
    # Generate recommendations
    recommendations = []
    
    for food in target_foods.index:
        try:
            # Prepare features for this food pair
            features = prepare_features(input_food, food, food_df, original_df)
            
            # Apply scaling if scaler is available
            if scaler is not None:
                features = scaler.transform(features.reshape(1, -1))[0]
            
            # Get prediction
            score = model.predict([features])[0]
            
            # Add to recommendations
            recommendations.append((food, score))
        except Exception as e:
            print(f"Error predicting for '{food}': {str(e)}", file=sys.stderr)
    
    # Sort by score (highest first)
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Prepare result
    result = {
        "input_food": input_food,
        "input_type": original_df.loc[input_food, 'type'],
        "target_type": food_type,
        "recommendations": [{"food": food, "score": float(score)} for food, score in recommendations[:top_n]]
    }
    
    return result

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Food matching recommendation system')
    parser.add_argument('input_food', type=str, help='Name of the input food')
    parser.add_argument('food_type', type=str, help='Type of food to recommend')
    parser.add_argument('--model', type=str, help='Path to the model file', 
                       default='model.pkl')
    parser.add_argument('--top', type=int, help='Number of recommendations', 
                       default=3)
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) >= 3:
        # Check if using argparse or direct arguments
        if '--' in ' '.join(sys.argv):
            args = parse_args()
            input_food = args.input_food
            food_type = args.food_type
            model_path = args.model
            top_n = args.top
        else:
            # Simple mode with just 2 arguments
            input_food = sys.argv[1]
            food_type = sys.argv[2]
            
            # Use default model path: search in current directory or up one level
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = 'modelnew_pickle.pkl'  # First try current directory
            
            if not os.path.exists(model_path):
                # Try project root directory
                model_path = os.path.join(base_dir, 'modelnew_pickle.pkl')
            
            # # If still not found, try a few other common paths
            # if not os.path.exists(model_path):
            #     model_paths = [
            #         os.path.join(base_dir, 'model/modeln.pkl'),
            #         os.path.join(base_dir, 'models/model.pkl'),
            #         os.path.join(base_dir, 'Project-AI-service-dev/model.pkl'),
            #         os.path.join(base_dir, 'Project-AI-service-dev/model2.pkl')
            #     ]
                
            #     for path in model_paths:
            #         if os.path.exists(path):
            #             model_path = path
            #             break
            
            top_n = 3  # Default
    else:
        print(json.dumps({"error": "Usage: python match.py <input_food> <food_type>"}))
        sys.exit(1)
    
    try:
        # Load the model
        print(f"Loading model from {model_path}", file=sys.stderr)
        model_data = load_model(model_path)
        
        # Get recommendations
        result = recommend_by_type(model_data, input_food, food_type, top_n)
        
        # Print as JSON (will be captured by the calling process)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)