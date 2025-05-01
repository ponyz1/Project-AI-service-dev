# match.py
import os
import pickle
import numpy as np
import sys
import json
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'model.pkl')

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
food_df = model_data['food_data']
original_df = model_data['original_food_data']

def recommend_by_type(input_food, food_type, top_n=3):
    if input_food not in original_df.index:
        return {"error": f"ไม่พบเมนู '{input_food}'"}

    available_types = original_df['type'].unique()
    if food_type not in available_types:
        return {"error": f"ไม่พบประเภท '{food_type}'"}

    target_foods = original_df[(original_df['type'] == food_type) & (original_df.index != input_food)]
    if len(target_foods) == 0:
        return {"error": f"ไม่มีเมนูประเภท '{food_type}'"}

    input_features = food_df.loc[input_food].values
    recommendations = []

    for food in target_foods.index:
        target_features = food_df.loc[food].values
        combined_features = np.concatenate([input_features, target_features])
        score = model.predict([combined_features])[0]
        recommendations.append((food, score))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    return {
        "input_food": input_food,
        "input_type": original_df.loc[input_food, 'type'],
        "target_type": food_type,
        "recommendations": [{"food": food, "score": float(score)} for food, score in recommendations[:top_n]]
    }


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python match.py <input_food> <food_type>"}))
        sys.exit(1)

    input_food = sys.argv[1]
    food_type = sys.argv[2]
    result = recommend_by_type(input_food, food_type)
    print(json.dumps(result, ensure_ascii=False))
