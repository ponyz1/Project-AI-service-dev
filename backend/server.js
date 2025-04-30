const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const { recommend_by_type } = require('./match'); // เรียกใช้ฟังก์ชันจาก match.py (ผ่าน Node.js module)

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

let modelData = null;
try {
  modelData = require('../model_data.json');
} catch (err) {
  console.error(" ไม่สามารถโหลด model_data.json ได้:", err.message);
}


app.get('/api/foods', (req, res) => {
  if (!modelData) return res.status(500).json({ error: 'Model not loaded' });

  const searchTerm = req.query.search?.toLowerCase() || '';
  const foods = modelData.original_food_data;

  const filteredFoods = Object.keys(foods)
    .filter(name => name.toLowerCase().includes(searchTerm))
    .map(name => ({
      name,
      type: foods[name].type
    }));

  res.json(filteredFoods);
});


app.get('/api/food-types', (req, res) => {
  if (!modelData) return res.status(500).json({ error: 'Model not loaded' });

  res.json(modelData.food_types);
});


app.post('/api/match', async (req, res) => {
  const { food, foodType } = req.body;

  if (!food || !foodType) {
    return res.status(400).json({ error: 'Missing food or foodType' });
  }

  try {

    const result = await recommend_by_type(food, foodType);
    if (result.error) {
      return res.status(404).json({ error: result.error });
    }

    res.json(result);
  } catch (err) {
    console.error(' Python matching error:', err.message);
    res.status(500).json({ error: 'Failed to run food matching model' });
  }
});


if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, 'client/build')));

  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'client/build', 'index.html'));
  });
}

app.listen(PORT, () => {
  console.log(` Server running at http://localhost:${PORT}`);
});
