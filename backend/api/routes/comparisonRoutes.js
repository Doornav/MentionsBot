/**
 * Comparison Routes
 * Endpoints for comparing market-implied vs model-implied probabilities
 */

const express = require('express');
const router = express.Router();
const comparisonController = require('../controllers/comparisonController');

// Get all comparisons
router.get('/', comparisonController.getAllComparisons);

// Get comparison by ID
router.get('/:id', comparisonController.getComparisonById);

// Get comparisons by market
router.get('/market/:marketId', comparisonController.getComparisonsByMarket);

// Generate new comparison
router.post('/generate/:marketId', comparisonController.generateComparison);

// Get comparison analytics (divergence metrics, trends, etc.)
router.get('/:id/analytics', comparisonController.getComparisonAnalytics);

// Get comparison history
router.get('/:id/history', comparisonController.getComparisonHistory);

module.exports = router;