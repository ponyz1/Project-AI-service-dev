import pickle
import json

with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)


original_food_data_df = model_data["original_food_data"]
original_food_data_dict = original_food_data_df.to_dict(orient="index")


food_types = sorted(list(set(entry["type"] for entry in original_food_data_dict.values())))


output = {
    "original_food_data": original_food_data_dict,
    "food_types": food_types
}

with open("model_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(" model_data.json สร้างเรียบร้อยแล้ว")
