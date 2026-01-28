/**
 * Forecast Controller
 * Handles business logic for forecast-related operations
 */

const ForecastService = require('../../services/forecastService');
const forecastService = new ForecastService();

// Get all forecasts
exports.getAllForecasts = async (req, res) => {
  try {
    const forecasts = await forecastService.getAllForecasts();
    res.status(200).json(forecasts);
  } catch (error) {
    console.error('Error fetching forecasts:', error);
    res.status(500).json({ error: 'Failed to fetch forecasts' });
  }
};

// Get forecast by ID
exports.getForecastById = async (req, res) => {
  try {
    const forecast = await forecastService.getForecastById(req.params.id);
    if (!forecast) {
      return res.status(404).json({ error: 'Forecast not found' });
    }
    res.status(200).json(forecast);
  } catch (error) {
    console.error('Error fetching forecast:', error);
    res.status(500).json({ error: 'Failed to fetch forecast' });
  }
};

// Get forecasts by market
exports.getForecastsByMarket = async (req, res) => {
  try {
    const forecasts = await forecastService.getForecastsByMarket(req.params.marketId);
    res.status(200).json(forecasts);
  } catch (error) {
    console.error('Error fetching forecasts for market:', error);
    res.status(500).json({ error: 'Failed to fetch forecasts for market' });
  }
};

// Create new forecast
exports.createForecast = async (req, res) => {
  try {
    const forecast = await forecastService.createForecast(req.body);
    res.status(201).json(forecast);
  } catch (error) {
    console.error('Error creating forecast:', error);
    res.status(500).json({ error: 'Failed to create forecast' });
  }
};

// Update forecast
exports.updateForecast = async (req, res) => {
  try {
    const forecast = await forecastService.updateForecast(req.params.id, req.body);
    if (!forecast) {
      return res.status(404).json({ error: 'Forecast not found' });
    }
    res.status(200).json(forecast);
  } catch (error) {
    console.error('Error updating forecast:', error);
    res.status(500).json({ error: 'Failed to update forecast' });
  }
};

// Generate forecast using ML models
exports.generateForecast = async (req, res) => {
  try {
    const { marketId } = req.params;
    const { modelType, parameters } = req.body;
    
    const forecast = await forecastService.generateForecast(marketId, modelType, parameters);
    res.status(201).json(forecast);
  } catch (error) {
    console.error('Error generating forecast:', error);
    res.status(500).json({ error: 'Failed to generate forecast' });
  }
};

// Get forecast history
exports.getForecastHistory = async (req, res) => {
  try {
    const history = await forecastService.getForecastHistory(req.params.id);
    res.status(200).json(history);
  } catch (error) {
    console.error('Error fetching forecast history:', error);
    res.status(500).json({ error: 'Failed to fetch forecast history' });
  }
};