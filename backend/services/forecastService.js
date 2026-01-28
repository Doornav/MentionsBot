/**
 * Forecast Service
 * 
 * Handles forecast generation, management, and retrieval
 * using various model types based on market and data inputs.
 */

const TimeSeriesModel = require('../models/forecast/timeSeriesModel');
const MarketDataService = require('./marketDataService');
const EvidenceService = require('./evidenceService');

class ForecastService {
  constructor(config = {}) {
    this.config = {
      defaultModelType: 'timeSeries',
      defaultHorizon: 30, // days
      minimumDataPoints: 14, // Minimum data points required for forecasting
      ...config
    };
    
    this.marketDataService = new MarketDataService();
    this.evidenceService = new EvidenceService();
    this.models = {
      timeSeries: new TimeSeriesModel()
    };
    
    // In-memory store for forecasts (would be replaced by database in production)
    this.forecasts = [];
  }

  /**
   * Generate a new forecast for a market
   * @param {string} marketId - The market identifier
   * @param {string} modelType - Type of model to use
   * @param {Object} parameters - Model-specific parameters
   * @returns {Promise<Object>} - The generated forecast
   */
  async generateForecast(marketId, modelType = this.config.defaultModelType, parameters = {}) {
    try {
      // Get historical market data
      const historicalData = await this.marketDataService.getHistoricalData(marketId, {
        days: parameters.dataHistoryDays || 90,
        resolution: parameters.resolution || '1d'
      });
      
      // Get current market data
      const currentMarketData = await this.marketDataService.getMarketData(marketId);
      
      // Check if we have enough data points
      if (historicalData.dataPoints.length < this.config.minimumDataPoints) {
        throw new Error(`Insufficient data points (${historicalData.dataPoints.length}) for forecasting`);
      }
      
      // Get relevant evidence
      const evidence = await this.evidenceService.getEvidenceForMarket(marketId);
      
      // Generate forecast based on model type
      let forecastResult;
      
      switch (modelType.toLowerCase()) {
        case 'timeseries':
        default:
          forecastResult = await this.generateTimeSeriesForecast(
            historicalData, 
            currentMarketData,
            parameters
          );
          break;
      }
      
      // Incorporate evidence if available
      if (evidence && evidence.length > 0) {
        forecastResult = this.incorporateEvidence(forecastResult, evidence);
      }
      
      // Store the forecast
      const forecast = {
        id: this.generateForecastId(),
        marketId,
        modelType,
        timestamp: new Date().toISOString(),
        generatedProbability: forecastResult.probability,
        confidenceScore: forecastResult.confidence,
        forecastHorizon: parameters.horizon || this.config.defaultHorizon,
        probabilityDistribution: forecastResult.distribution || null,
        parameters: {
          ...parameters,
          dataPointsUsed: historicalData.dataPoints.length,
          evidenceCount: evidence ? evidence.length : 0
        },
        marketData: {
          currentProbability: currentMarketData.probability,
          timestamp: currentMarketData.timestamp
        },
        metadata: {
          evidenceIncorporated: !!evidence,
          methodology: forecastResult.methodology
        }
      };
      
      this.forecasts.push(forecast);
      return forecast;
    } catch (error) {
      console.error(`Error generating forecast for ${marketId}:`, error);
      throw new Error(`Failed to generate forecast: ${error.message}`);
    }
  }

  /**
   * Generate time series forecast
   * @private
   */
  async generateTimeSeriesForecast(historicalData, currentMarketData, parameters = {}) {
    const timeSeriesModel = this.models.timeSeries;
    
    // Extract probability time series from historical data
    const probabilities = historicalData.dataPoints.map(point => point.probability);
    
    // Train the model with historical data
    await timeSeriesModel.train(probabilities, {
      epochs: parameters.epochs || 50,
      batchSize: parameters.batchSize || 32
    });
    
    // Generate forecast
    const forecastValues = timeSeriesModel.forecast(probabilities);
    const probability = forecastValues[0]; // Take the first forecasted value
    
    // Calculate confidence based on model's training performance and data quality
    const confidence = 0.7; // Mock value, would be derived from model metrics
    
    return {
      probability,
      confidence,
      distribution: this.generateProbabilityDistribution(probability, confidence),
      methodology: 'Time series analysis with LSTM neural network'
    };
  }

  /**
   * Generate probability distribution around the point estimate
   * @private
   */
  generateProbabilityDistribution(probability, confidence) {
    // Standard deviation is inversely proportional to confidence
    const stdDev = (1 - confidence) * 0.2;
    
    // Generate 21 points representing the probability distribution
    const distribution = [];
    
    for (let i = -10; i <= 10; i++) {
      // Center point (x=0) represents the probability estimate
      const x = probability + (i * stdDev * 0.2);
      
      // Ensure the x value is within bounds
      const boundedX = Math.max(0, Math.min(1, x));
      
      // Generate normal distribution around probability
      const y = Math.exp(-0.5 * Math.pow(i/5, 2));
      
      distribution.push({ x: boundedX, y });
    }
    
    // Normalize y values to sum to 1
    const sum = distribution.reduce((total, point) => total + point.y, 0);
    return distribution.map(point => ({ 
      x: point.x, 
      y: point.y / sum 
    }));
  }

  /**
   * Incorporate evidence into the forecast
   * @private
   */
  incorporateEvidence(forecastResult, evidence) {
    // Simple evidence incorporation for now
    // Would be more sophisticated in production
    
    // Calculate evidence impact based on relevance and sentiment
    let evidenceImpact = 0;
    let totalWeight = 0;
    
    for (const item of evidence) {
      const weight = item.relevance * item.confidence;
      
      // Convert sentiment (-1 to 1) to probability impact
      const impact = item.sentiment * 0.05; // Max 5% shift per evidence item
      
      evidenceImpact += impact * weight;
      totalWeight += weight;
    }
    
    // Normalize by total weight
    if (totalWeight > 0) {
      evidenceImpact /= totalWeight;
    }
    
    // Apply evidence impact to probability (capped)
    const adjustedProbability = Math.max(0, Math.min(1, 
      forecastResult.probability + evidenceImpact
    ));
    
    // Evidence can slightly boost confidence if aligned with model
    const sentimentAlignment = 1 - Math.min(
      Math.abs(evidenceImpact) / 0.05, 
      1
    );
    
    const adjustedConfidence = Math.min(
      forecastResult.confidence + (sentimentAlignment * 0.1), 
      0.95
    );
    
    return {
      ...forecastResult,
      probability: adjustedProbability,
      confidence: adjustedConfidence,
      evidenceImpact
    };
  }

  /**
   * Generate a unique forecast ID
   * @private
   */
  generateForecastId() {
    return `FC-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Get all forecasts
   */
  async getAllForecasts() {
    return this.forecasts;
  }

  /**
   * Get forecast by ID
   */
  async getForecastById(id) {
    return this.forecasts.find(forecast => forecast.id === id);
  }

  /**
   * Get forecasts for a specific market
   */
  async getForecastsByMarket(marketId) {
    return this.forecasts.filter(forecast => forecast.marketId === marketId);
  }

  /**
   * Create a forecast manually
   */
  async createForecast(forecastData) {
    const forecast = {
      id: this.generateForecastId(),
      timestamp: new Date().toISOString(),
      ...forecastData
    };
    
    this.forecasts.push(forecast);
    return forecast;
  }

  /**
   * Update a forecast
   */
  async updateForecast(id, updateData) {
    const forecastIndex = this.forecasts.findIndex(forecast => forecast.id === id);
    
    if (forecastIndex === -1) {
      return null;
    }
    
    this.forecasts[forecastIndex] = {
      ...this.forecasts[forecastIndex],
      ...updateData,
      lastUpdated: new Date().toISOString()
    };
    
    return this.forecasts[forecastIndex];
  }

  /**
   * Get forecast history
   */
  async getForecastHistory(id) {
    // In a real implementation, this would retrieve the version history
    // For now, just return the current forecast
    const forecast = await this.getForecastById(id);
    
    if (!forecast) {
      return null;
    }
    
    // Mock history data
    return {
      forecastId: id,
      history: [
        {
          timestamp: forecast.timestamp,
          probability: forecast.generatedProbability,
          confidence: forecast.confidenceScore
        }
      ]
    };
  }
}

class EvidenceService {
  // Mock implementation
  async getEvidenceForMarket(marketId) {
    return [
      {
        id: 'EV-1',
        title: 'Mock Evidence Item',
        relevance: 0.8,
        confidence: 0.7,
        sentiment: 0.5, // Range: -1 to 1, where 1 is strongly positive
        timestamp: new Date().toISOString()
      }
    ];
  }
}

module.exports = ForecastService;