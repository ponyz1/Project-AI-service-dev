from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
from typing import List, Optional

# Import the function from match.py
# Assuming match.py is in the same directory and has a function called recommend_by_type
from match import recommend_by_type

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model data
model_data = None
try:
    with open("../model_data.json", "r", encoding="utf-8") as f:
        model_data = json.load(f)
except Exception as e:
    print(f"ไม่สามารถโหลด model_data.json ได้: {str(e)}")

# Define response models
class FoodItem(BaseModel):
    name: str
    type: str

class MatchRequest(BaseModel):
    food: str
    foodType: str

# API routes
@app.get("/api/foods", response_model=List[FoodItem])
async def get_foods(search: Optional[str] = Query(None)):
    if not model_data:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    search_term = search.lower() if search else ''
    foods = model_data["original_food_data"]
    
    filtered_foods = [
        {"name": name, "type": foods[name]["type"]}
        for name in foods
        if search_term in name.lower()
    ]
    
    return filtered_foods

@app.get("/api/food-types")
async def get_food_types():
    if not model_data:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return model_data["food_types"]

@app.post("/api/match")
async def match_foods(request: MatchRequest):
    if not request.food or not request.foodType:
        raise HTTPException(status_code=400, detail="Missing food or foodType")
    
    try:
        result = await recommend_by_type(request.food, request.foodType)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    except Exception as e:
        print(f"Python matching error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to run food matching model")

# Serve static files in production
if os.environ.get("ENVIRONMENT") == "production":
    app.mount("/", StaticFiles(directory="client/build", html=True), name="static")
    
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        return FileResponse("client/build/index.html")

# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)