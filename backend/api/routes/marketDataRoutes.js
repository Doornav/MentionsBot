/**
 * Market Data Routes
 * Endpoints for retrieving and processing market-implied probabilities
 */

const express = require('express');
const router = express.Router();
const marketDataController = require('../controllers/marketDataController');

// Get current market data for all markets
router.get('/', marketDataController.getAllMarketData);

// Get market data by ID
router.get('/:id', marketDataController.getMarketDataById);

// Get market data for a specific vertical (e.g., US politics, macro)
router.get('/vertical/:vertical', marketDataController.getMarketDataByVertical);

// Get historical market data
router.get('/:id/history', marketDataController.getMarketDataHistory);

// Refresh market data from external sources
router.post('/refresh', marketDataController.refreshMarketData);

// Add new market data source
router.post('/sources', marketDataController.addMarketDataSource);

module.exports = router;