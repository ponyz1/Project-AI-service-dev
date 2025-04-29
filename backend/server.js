const express = require('express');
const cors = require('cors');
const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.get('/', (req, res) => {
  res.json({
    status: 'success',
    message: 'Welcome to our API!',
    endpoints: {
      test: 'POST /api/test'
    }
  });
});

// Test route
app.post('/test', (req, res) => {
  res.json({
    status: 'success',
    message: 'Backend is working!',
    data: req.body,
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});