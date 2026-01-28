/**
 * API Routes Index
 * Main router that combines all API route modules
 */

const express = require('express');
const router = express.Router();

// Import route modules
const forecastRoutes = require('./forecastRoutes');
const marketDataRoutes = require('./marketDataRoutes');
const comparisonRoutes = require('./comparisonRoutes');
const evidenceRoutes = require('./evidenceRoutes');

// Register route modules
router.use('/forecasts', forecastRoutes);
router.use('/market-data', marketDataRoutes);
router.use('/comparisons', comparisonRoutes);
router.use('/evidence', evidenceRoutes);

module.exports = router;