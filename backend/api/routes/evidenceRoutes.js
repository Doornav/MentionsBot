/**
 * Evidence Routes
 * Endpoints for retrieving and managing evidence that supports forecasts
 */

const express = require('express');
const router = express.Router();
const evidenceController = require('../controllers/evidenceController');

// Get all evidence
router.get('/', evidenceController.getAllEvidence);

// Get evidence by ID
router.get('/:id', evidenceController.getEvidenceById);

// Get evidence for a specific forecast
router.get('/forecast/:forecastId', evidenceController.getEvidenceByForecast);

// Get evidence by category
router.get('/category/:category', evidenceController.getEvidenceByCategory);

// Add new evidence
router.post('/', evidenceController.createEvidence);

// Update evidence
router.put('/:id', evidenceController.updateEvidence);

// Generate evidence from sources
router.post('/generate/:forecastId', evidenceController.generateEvidence);

// Rate evidence importance
router.post('/:id/rate', evidenceController.rateEvidence);

module.exports = router;