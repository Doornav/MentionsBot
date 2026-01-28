/**
 * Forecast Routes
 * Endpoints for model-generated forecasts and probability distributions
 */

const express = require('express');
const router = express.Router();
const forecastController = require('../controllers/forecastController');

// Get all forecasts
router.get('/', forecastController.getAllForecasts);

// Get forecast by ID
router.get('/:id', forecastController.getForecastById);

// Get forecasts for a specific market
router.get('/market/:marketId', forecastController.getForecastsByMarket);

// Create new forecast
router.post('/', forecastController.createForecast);

// Update forecast
router.put('/:id', forecastController.updateForecast);

// Generate forecast from model
router.post('/generate/:marketId', forecastController.generateForecast);

// Get forecast history/timeline
router.get('/:id/history', forecastController.getForecastHistory);

module.exports = router;