/**
 * Market Data Service
 * 
 * Handles the retrieval and processing of market-implied probabilities
 * from various prediction markets and exchanges.
 */

const axios = require('axios');
const DataProcessor = require('../data/processors/dataProcessor');

class MarketDataService {
  constructor(config = {}) {
    this.config = {
      refreshInterval: 15 * 60 * 1000, // 15 minutes default
      cacheTTL: 5 * 60 * 1000, // 5 minutes default
      sources: ['manifold', 'polymarket', 'kalshi', 'metaculus'], // Default sources
      ...config
    };
    
    this.dataProcessor = new DataProcessor();
    this.cache = new Map();
    this.lastFetchTime = {};
  }

  /**
   * Get market data for a specific market
   * @param {string} marketId - The market identifier
   * @param {Object} options - Additional options
   * @returns {Promise<Object>} - Market data
   */
  async getMarketData(marketId, options = {}) {
    const defaultOptions = {
      forceRefresh: false,
      source: null,
    };
    
    const opts = { ...defaultOptions, ...options };
    
    // Check cache if not forcing a refresh
    if (!opts.forceRefresh) {
      const cachedData = this.cache.get(marketId);
      if (cachedData && Date.now() - cachedData.timestamp < this.config.cacheTTL) {
        return cachedData;
      }
    }
    
    // Determine which source to use
    const source = opts.source || this.identifySourceFromMarketId(marketId);
    
    // Fetch from the appropriate source
    try {
      const marketData = await this.fetchFromSource(source, marketId);
      
      // Process the raw data
      const processedData = this.dataProcessor.processMarketData(marketData, source);
      
      // Cache the result
      this.cache.set(marketId, {
        ...processedData,
        timestamp: Date.now()
      });
      
      this.lastFetchTime[marketId] = Date.now();
      
      return processedData;
    } catch (error) {
      console.error(`Error fetching market data for ${marketId}:`, error);
      throw new Error(`Failed to fetch market data: ${error.message}`);
    }
  }

  /**
   * Identify the source platform from the market ID format
   * @private
   */
  identifySourceFromMarketId(marketId) {
    if (marketId.startsWith('MF-')) {
      return 'manifold';
    } else if (marketId.startsWith('PM-')) {
      return 'polymarket';
    } else if (marketId.startsWith('KL-')) {
      return 'kalshi';
    } else if (marketId.startsWith('MC-')) {
      return 'metaculus';
    }
    
    // Default to first configured source
    return this.config.sources[0];
  }

  /**
   * Fetch market data from a specific source
   * @private
   */
  async fetchFromSource(source, marketId) {
    // Extract the native ID from our prefixed ID
    const nativeId = marketId.split('-')[1];
    
    switch (source.toLowerCase()) {
      case 'manifold':
        return this.fetchFromManifold(nativeId);
        
      case 'polymarket':
        return this.fetchFromPolymarket(nativeId);
        
      case 'kalshi':
        return this.fetchFromKalshi(nativeId);
        
      case 'metaculus':
        return this.fetchFromMetaculus(nativeId);
        
      default:
        throw new Error(`Unsupported market source: ${source}`);
    }
  }

  /**
   * Fetch from Manifold Markets API
   * @private
   */
  async fetchFromManifold(nativeId) {
    try {
      const response = await axios.get(`https://api.manifold.markets/v0/market/${nativeId}`);
      
      if (!response.data) {
        throw new Error('No data returned from Manifold API');
      }
      
      // Transform to standard format
      return {
        id: `MF-${nativeId}`,
        title: response.data.question,
        description: response.data.description,
        probability: response.data.probability,
        volume: response.data.volume,
        liquidity: response.data.liquidity,
        closeTime: response.data.closeTime,
        source: 'manifold',
        url: `https://manifold.markets/${response.data.creatorUsername}/${nativeId}`,
        rawData: response.data
      };
    } catch (error) {
      console.error('Manifold API error:', error);
      throw new Error(`Manifold API error: ${error.message}`);
    }
  }

  /**
   * Fetch from Polymarket API
   * @private
   */
  async fetchFromPolymarket(nativeId) {
    // This would be replaced with actual API implementation
    // Mock implementation for now
    return {
      id: `PM-${nativeId}`,
      title: 'Mock Polymarket Question',
      description: 'This is a mock implementation',
      probability: 0.65,
      volume: 120000,
      liquidity: 45000,
      closeTime: new Date().getTime() + 7 * 24 * 60 * 60 * 1000, // 7 days from now
      source: 'polymarket',
      url: `https://polymarket.com/event/${nativeId}`,
      rawData: {}
    };
  }

  /**
   * Fetch from Kalshi API
   * @private
   */
  async fetchFromKalshi(nativeId) {
    // This would be replaced with actual API implementation
    // Mock implementation for now
    return {
      id: `KL-${nativeId}`,
      title: 'Mock Kalshi Question',
      description: 'This is a mock implementation',
      probability: 0.35,
      volume: 85000,
      liquidity: 25000,
      closeTime: new Date().getTime() + 14 * 24 * 60 * 60 * 1000, // 14 days from now
      source: 'kalshi',
      url: `https://kalshi.com/markets/${nativeId}`,
      rawData: {}
    };
  }

  /**
   * Fetch from Metaculus API
   * @private
   */
  async fetchFromMetaculus(nativeId) {
    // This would be replaced with actual API implementation
    // Mock implementation for now
    return {
      id: `MC-${nativeId}`,
      title: 'Mock Metaculus Question',
      description: 'This is a mock implementation',
      probability: 0.72,
      // Metaculus specific fields
      communityPrediction: 0.72,
      predictionCount: 156,
      closeTime: new Date().getTime() + 30 * 24 * 60 * 60 * 1000, // 30 days from now
      source: 'metaculus',
      url: `https://www.metaculus.com/questions/${nativeId}`,
      rawData: {}
    };
  }

  /**
   * Get market data for all markets in a vertical
   * @param {string} vertical - Vertical category (e.g., 'politics', 'crypto')
   * @returns {Promise<Array>} - List of market data
   */
  async getMarketsByVertical(vertical) {
    try {
      // This would be replaced with actual API calls
      // Mock implementation for now
      const mockMarkets = [
        {
          id: 'MF-123456',
          title: 'Will candidate X win the election?',
          probability: 0.65,
          vertical: 'politics'
        },
        {
          id: 'PM-789012',
          title: 'Will legislation Y pass this year?',
          probability: 0.22,
          vertical: 'politics'
        },
        {
          id: 'KL-345678',
          title: 'Will inflation exceed 3% next quarter?',
          probability: 0.48,
          vertical: 'economics'
        }
      ];
      
      return mockMarkets.filter(market => 
        market.vertical.toLowerCase() === vertical.toLowerCase()
      );
    } catch (error) {
      console.error(`Error fetching markets for vertical ${vertical}:`, error);
      throw new Error(`Failed to fetch markets: ${error.message}`);
    }
  }

  /**
   * Get historical market data for a specific market
   * @param {string} marketId - The market identifier
   * @param {Object} options - Additional options
   * @returns {Promise<Array>} - Historical data points
   */
  async getHistoricalData(marketId, options = {}) {
    const defaultOptions = {
      days: 30,
      resolution: '1d' // Options: 1h, 1d, 1w
    };
    
    const opts = { ...defaultOptions, ...options };
    
    try {
      // This would be replaced with actual API implementation
      // Mock implementation for now
      const source = this.identifySourceFromMarketId(marketId);
      const nativeId = marketId.split('-')[1];
      
      // Generate mock historical data
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - opts.days);
      
      const dataPoints = [];
      let currentDate = new Date(startDate);
      
      // Generate a price trajectory with some randomness
      let currentPrice = Math.random() * 0.5 + 0.25; // Start between 0.25 and 0.75
      
      // Add a trend bias
      const trendDirection = Math.random() > 0.5 ? 1 : -1;
      const trendStrength = Math.random() * 0.01; // Daily trend strength
      
      while (currentDate <= endDate) {
        // Add some random walk + trend
        const noise = (Math.random() - 0.5) * 0.03; // Daily noise
        currentPrice += noise + (trendDirection * trendStrength);
        
        // Ensure bounds
        currentPrice = Math.max(0.01, Math.min(0.99, currentPrice));
        
        dataPoints.push({
          timestamp: currentDate.toISOString(),
          probability: currentPrice,
          volume: Math.floor(Math.random() * 10000) + 1000
        });
        
        // Increment date based on resolution
        switch (opts.resolution) {
          case '1h':
            currentDate.setHours(currentDate.getHours() + 1);
            break;
          case '1w':
            currentDate.setDate(currentDate.getDate() + 7);
            break;
          case '1d':
          default:
            currentDate.setDate(currentDate.getDate() + 1);
        }
      }
      
      return {
        marketId,
        source,
        resolution: opts.resolution,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
        dataPoints
      };
    } catch (error) {
      console.error(`Error fetching historical data for ${marketId}:`, error);
      throw new Error(`Failed to fetch historical data: ${error.message}`);
    }
  }

  /**
   * Refresh all cached market data
   */
  async refreshAllMarketData() {
    const marketIds = Array.from(this.cache.keys());
    
    const refreshPromises = marketIds.map(marketId => 
      this.getMarketData(marketId, { forceRefresh: true })
    );
    
    await Promise.all(refreshPromises);
    console.log(`Refreshed ${marketIds.length} markets`);
    
    return marketIds.length;
  }
}

module.exports = MarketDataService;