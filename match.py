import pickle
import numpy as np
import pandas as pd

# โหลดโมเดลและข้อมูล
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
food_df = model_data['food_data']
original_df = model_data['original_food_data']

def recommend_by_type(input_food, food_type, top_n=3):
    """
    แนะนำอาหารที่เข้ากันดีกับเมนูที่เลือก โดยกรองตามประเภท
    Parameters:
    - input_food: ชื่ออาหารที่เลือก
    - food_type: ประเภทอาหารที่ต้องการ match
    - top_n: จำนวนคำแนะนำที่ต้องการ
    """
    # ตรวจสอบเมนูที่เลือก
    if input_food not in original_df.index:
        return {"error": f"ไม่พบเมนู '{input_food}' ในระบบ"}
    
    # ตรวจสอบประเภทอาหาร
    available_types = original_df['type'].unique()
    if food_type not in available_types:
        return {"error": f"ไม่พบประเภท '{food_type}' ในระบบ (ประเภทที่มี: {', '.join(available_types)})"}
    
    # กรองเมนูเฉพาะประเภทที่ต้องการ (ไม่รวมเมนูที่เลือก)
    target_foods = original_df[(original_df['type'] == food_type) & (original_df.index != input_food)]
    
    if len(target_foods) == 0:
        return {"error": f"ไม่มีเมนูประเภท '{food_type}' ในระบบ"}
    
    # คำนวณคะแนนความเข้ากัน
    recommendations = []
    input_features = food_df.loc[input_food].values
    
    for food in target_foods.index:
        target_features = food_df.loc[food].values
        combined_features = np.concatenate([input_features, target_features])
        score = model.predict([combined_features])[0]
        recommendations.append((food, score))
    
    # เรียงลำดับตามคะแนน
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # เตรียมผลลัพธ์
    result = {
        "input_food": input_food,
        "input_type": original_df.loc[input_food, 'type'],
        "target_type": food_type,
        "recommendations": [{"food": food, "score": float(score)} for food, score in recommendations[:top_n]]
    }
    
    return result

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    print("\n=== ระบบแนะนำคู่เมนูอาหาร ===")
    print("\nเมนูที่มีในระบบ:")
    print(original_df.index.tolist())
    print("\nประเภทอาหารที่มี:")
    print(original_df['type'].unique())
    
    while True:
        print("\n" + "="*50)
        input_food = input("\nป้อนชื่อเมนู: ").strip()
        if input_food.lower() == 'exit':
            break
            
        food_type = input("ป้อนประเภทอาหารที่ต้องการ match: ").strip()
        
        result = recommend_by_type(input_food, food_type)
        
        if "error" in result:
            print("\n ข้อผิดพลาด:", result["error"])
        else:
            print(f"\n เมนู '{result['input_food']}' ({result['input_type']}) เข้ากันดีกับ:")
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"{i}. {rec['food']} (คะแนน: {rec['score']:.2f})")