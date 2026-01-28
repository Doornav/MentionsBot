/**
 * Market Insights Engine - Backend Entry Point
 * 
 * This is the main entry point for the ML backend service that powers the
 * market insights engine. This service provides:
 * 
 * 1. Model-implied probability calculations
 * 2. Market vs Model probability comparisons
 * 3. Evidence evaluation and confidence scoring
 * 4. Forecast generation and updates
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const apiRoutes = require('./api/routes');

// Initialize express app
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// API Routes
app.use('/api', apiRoutes);

// Basic health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ 
    status: 'ok',
    service: 'market-insights-engine-backend',
    version: '0.1.0'
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Market Insights Engine Backend running on port ${PORT}`);
});