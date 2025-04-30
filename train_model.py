import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

# โหลดข้อมูล
food_df = pd.read_csv('food_data.csv')
match_df = pd.read_csv('match_rating.csv')

# แยกคอลัมน์ที่เป็นตัวเลขและประเภท
numeric_cols = ['spicy', 'sweet', 'sour', 'salty', 'texture']
categorical_col = 'type'

# One-Hot Encoding สำหรับคอลัมน์ type
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
type_encoded = encoder.fit_transform(food_df[[categorical_col]])
encoded_cols = encoder.get_feature_names_out([categorical_col])

# สร้าง DataFrame ใหม่ที่รวมข้อมูลตัวเลขและ encoded
food_features = pd.DataFrame(
    np.hstack([food_df[numeric_cols].values, type_encoded]),
    columns=numeric_cols + list(encoded_cols),
    index=food_df['name']
)

food_df_processed = food_features

# สร้าง dataset สำหรับ training
X = []
y = []
pair_names = []

for _, row in match_df.iterrows():
    food1 = row['food1']
    food2 = row['food2']
    rating = row['rating']

    if food1 in food_df_processed.index and food2 in food_df_processed.index:
        vec1 = food_df_processed.loc[food1].values
        vec2 = food_df_processed.loc[food2].values
        combined = np.concatenate([vec1, vec2])
        X.append(combined)
        y.append(rating)
        pair_names.append((food1, food2))

if len(X) == 0:
    raise ValueError("ไม่มีคู่อาหารที่ตรงกันระหว่าง food_data และ match_rating")

X = np.array(X)
y = np.array(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and data
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'food_data': food_df_processed,
        'original_food_data': food_df.set_index('name'),
        'encoder': encoder,
        'numeric_cols': numeric_cols,
        'categorical_col': categorical_col
    }, f)

print('✅ Model trained and saved to model.pkl')
print(f'จำนวนคู่อาหารที่ใช้ฝึก: {len(X)}')
print(f'ประเภทอาหารทั้งหมด: {encoder.categories_[0].tolist()}')