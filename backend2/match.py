import os
import pickle
import numpy as np
from typing import Dict, List, Any, Optional

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'model.pkl')

try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    food_df = model_data['food_data']
    original_df = model_data['original_food_data']
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_data = None
    model = None
    food_df = None
    original_df = None

async def recommend_by_type(input_food: str, food_type: str, top_n: int = 3) -> Dict[str, Any]:
    if model is None or food_df is None or original_df is None:
        return {"error": "Model not loaded properly"}
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
        score = float(model.predict([combined_features])[0]) 
        recommendations.append((food, score))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return {
        "original_food": {
            "name": input_food,
            "type": original_df.loc[input_food, 'type']
        },
        "matched_foods": [
            {
                "name": food,
                "score": score,
                "type": food_type
            } 
            for food, score in recommendations[:top_n]
        ]
    }

if __name__ == "__main__":
    import asyncio
    
    async def test():
        result = await recommend_by_type("ต้มยำกุ้ง", "dessert")
        print(result)
    
    asyncio.run(test())