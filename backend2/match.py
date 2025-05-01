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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_model(model_path):
    try:
        model_data = joblib.load(model_path)
        print(f"Model loaded successfully with joblib from {model_path}", file=sys.stderr)
    except:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"Model loaded successfully with pickle from {model_path}", file=sys.stderr)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    return model_data

def prepare_features(input_food, target_food, food_data, original_food_data=None):
    if original_food_data is not None and isinstance(food_data, pd.DataFrame):
        flavor_cols = ['spicy', 'sweet', 'sour', 'salty', 'texture']
        if input_food not in food_data.index:
            raise ValueError(f"Food '{input_food}' not found in dataset")
        if target_food not in food_data.index:
            raise ValueError(f"Food '{target_food}' not found in dataset")
        food1_features = food_data.loc[input_food].values
        food2_features = food_data.loc[target_food].values

        if original_food_data is not None:
            food1_type = original_food_data.loc[input_food, 'type']
            food2_type = original_food_data.loc[target_food, 'type']
        else:
            food1_type = ""
            food2_type = ""
        
        diff = np.abs(food1_features - food2_features)
        product = food1_features * food2_features
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
        food1_features = food_data.loc[input_food].values
        food2_features = food_data.loc[target_food].values
        return np.concatenate([food1_features, food2_features])
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'modelnew_pickle.pkl')

if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(base_dir), 'modelnew_pickle.pkl')

try:
    MODEL_DATA = load_model(model_path)
    print(f"Model loaded successfully at module level", file=sys.stderr)
except Exception as e:
    print(f"Failed to load model at module level: {str(e)}", file=sys.stderr)
    MODEL_DATA = None

async def recommend_by_type(input_food, food_type, top_n=3):
    model_data = MODEL_DATA
    if model_data is None:
        try:
            model_data = load_model(model_path)
        except Exception as e:
            return {"error": f"Model not loaded: {str(e)}"}
    model = model_data['model']
    
    if 'food_data' in model_data and 'original_food_data' in model_data:
        food_df = model_data['food_data']
        original_df = model_data['original_food_data']
    elif 'food_data' in model_data:
        food_df = model_data['food_data']
        original_df = model_data['food_data']
    else:
        return {"error": "Invalid model data format. Missing food_data."}
    scaler = model_data.get('scaler', None)
    if input_food not in original_df.index:
        return {"error": f"ไม่พบเมนู '{input_food}'"}
    available_types = original_df['type'].unique()
    if food_type not in available_types:
        return {"error": f"ไม่พบประเภท '{food_type}'"}
    target_foods = original_df[(original_df['type'] == food_type) & (original_df.index != input_food)]
    if len(target_foods) == 0:
        return {"error": f"ไม่มีเมนูประเภท '{food_type}'"}
    recommendations = []
    
    for food in target_foods.index:
        try:
            features = prepare_features(input_food, food, food_df, original_df)
            if scaler is not None:
                features = scaler.transform(features.reshape(1, -1))[0]
            score = model.predict([features])[0]
            recommendations.append((food, score))
        except Exception as e:
            print(f"Error predicting for '{food}': {str(e)}", file=sys.stderr)
    recommendations.sort(key=lambda x: x[1], reverse=True)
    result = {
        "input_food": input_food,
        "input_type": original_df.loc[input_food, 'type'],
        "target_type": food_type,
        "recommendations": [{"food": food, "score": float(score)} for food, score in recommendations[:top_n]]
    }
    
    return result