/**
 * Data Transformers
 * 
 * Utility functions for transforming data between different formats
 * and structures needed by the application.
 */

/**
 * Transform market data from backend format to frontend format
 * @param {Object} marketData - Market data from backend
 * @returns {Object} - Formatted for frontend use
 */
function transformMarketForFrontend(marketData) {
  // Base transformation
  const transformed = {
    id: marketData.id,
    title: marketData.title,
    description: marketData.description || '',
    currentProbability: marketData.probability,
    resolveDate: marketData.closeTime,
    source: {
      name: marketData.source,
      url: marketData.url || ''
    },
    stats: {
      volume: marketData.volume || 0,
      liquidity: marketData.liquidity || 0,
      impliedOdds: marketData.impliedOdds || (1 / marketData.probability).toFixed(2)
    },
    lastUpdated: marketData.timestamp || marketData.processedAt || new Date().toISOString()
  };
  
  // Add additional fields if they exist
  if (marketData.vertical) {
    transformed.category = marketData.vertical;
  }
  
  if (marketData.tags) {
    transformed.tags = marketData.tags;
  }
  
  // Format different probability displays
  transformed.display = {
    probabilityPercent: `${(marketData.probability * 100).toFixed(1)}%`,
    impliedOdds: `${(1 / marketData.probability).toFixed(2)}`,
    decimalOdds: marketData.probability.toFixed(2)
  };
  
  return transformed;
}

/**
 * Transform forecast data from backend format to frontend format
 * @param {Object} forecastData - Forecast data from backend
 * @returns {Object} - Formatted for frontend use
 */
function transformForecastForFrontend(forecastData) {
  return {
    id: forecastData.id,
    marketId: forecastData.marketId,
    modelProbability: forecastData.generatedProbability,
    confidence: forecastData.confidenceScore,
    timestamp: forecastData.timestamp,
    horizon: forecastData.forecastHorizon,
    modelType: forecastData.modelType,
    display: {
      probabilityPercent: `${(forecastData.generatedProbability * 100).toFixed(1)}%`,
      confidencePercent: `${(forecastData.confidenceScore * 100).toFixed(0)}%`,
      impliedOdds: `${(1 / forecastData.generatedProbability).toFixed(2)}`
    },
    distribution: forecastData.probabilityDistribution || null,
    methodology: forecastData.metadata?.methodology || "AI-powered forecasting"
  };
}

/**
 * Transform comparison data from backend format to frontend format
 * @param {Object} comparisonData - Comparison data from backend
 * @returns {Object} - Formatted for frontend use
 */
function transformComparisonForFrontend(comparisonData) {
  const edge = comparisonData.comparison.edge || {};
  const metrics = comparisonData.comparison || {};
  
  return {
    id: comparisonData.id || `CMP-${Date.now()}`,
    market: transformMarketForFrontend(comparisonData.market),
    model: {
      probability: comparisonData.model.probability,
      confidence: comparisonData.model.confidence,
      display: {
        probabilityPercent: `${(comparisonData.model.probability * 100).toFixed(1)}%`
      }
    },
    divergence: {
      absolute: metrics.absoluteDivergence,
      relative: metrics.relativeDivergence,
      direction: metrics.divergenceDirection > 0 ? 'model-higher' : 'market-higher',
      significance: metrics.divergenceSignificance,
      display: {
        absolutePercent: `${(metrics.absoluteDivergence * 100).toFixed(1)}%`,
        relativePercent: `${(metrics.relativeDivergence * 100).toFixed(1)}%`,
        direction: metrics.divergenceDirection > 0 ? '↑' : '↓'
      }
    },
    edge: {
      raw: edge.rawEdge || 0,
      adjusted: edge.adjustedEdge || 0,
      confidence: edge.confidenceScore || 0,
      kelly: edge.kellyFraction || 0,
      recommendation: edge.recommendation || {
        action: 'HOLD',
        strength: 'neutral',
        reasoning: 'Insufficient data for recommendation'
      },
      display: {
        rawEdgePercent: `${((edge.rawEdge || 0) * 100).toFixed(1)}%`,
        adjustedEdgePercent: `${((edge.adjustedEdge || 0) * 100).toFixed(1)}%`
      }
    },
    summary: comparisonData.comparison.summary || 'No summary available',
    timestamp: comparisonData.timestamp
  };
}

/**
 * Transform evidence data from backend format to frontend format
 * @param {Object} evidenceData - Evidence data from backend
 * @returns {Object} - Formatted for frontend use
 */
function transformEvidenceForFrontend(evidenceData) {
  return {
    id: evidenceData.id,
    title: evidenceData.title,
    content: evidenceData.content || '',
    source: evidenceData.source || {
      name: 'Unknown',
      url: ''
    },
    metrics: {
      relevance: evidenceData.relevance || 0,
      sentiment: evidenceData.sentiment || 0,
      confidence: evidenceData.confidence || 0
    },
    display: {
      relevancePercent: `${((evidenceData.relevance || 0) * 100).toFixed(0)}%`,
      sentimentClass: getSentimentClass(evidenceData.sentiment || 0),
      confidenceBars: getConfidenceBars(evidenceData.confidence || 0)
    },
    timestamp: evidenceData.timestamp,
    category: evidenceData.category || 'general',
    tags: evidenceData.tags || []
  };
}

/**
 * Transform historical data from backend format to frontend chart format
 * @param {Object} historicalData - Historical data from backend
 * @returns {Object} - Formatted for chart library use
 */
function transformHistoricalDataForCharts(historicalData) {
  // Format for use with chart libraries like Chart.js or Recharts
  const timestamps = historicalData.dataPoints.map(point => point.timestamp);
  const probabilities = historicalData.dataPoints.map(point => point.probability);
  const volumes = historicalData.dataPoints.map(point => point.volume || 0);
  
  return {
    labels: timestamps,
    datasets: [
      {
        id: 'probability',
        label: 'Probability',
        data: probabilities,
        yAxisID: 'probability'
      },
      {
        id: 'volume',
        label: 'Volume',
        data: volumes,
        yAxisID: 'volume'
      }
    ],
    metadata: {
      marketId: historicalData.marketId,
      source: historicalData.source,
      resolution: historicalData.resolution,
      startDate: historicalData.startDate,
      endDate: historicalData.endDate
    }
  };
}

/**
 * Helper function to get sentiment class based on sentiment value
 * @private
 */
function getSentimentClass(sentiment) {
  if (sentiment > 0.3) return 'positive';
  if (sentiment < -0.3) return 'negative';
  return 'neutral';
}

/**
 * Helper function to convert confidence to visual representation
 * @private
 */
function getConfidenceBars(confidence) {
  // Convert confidence (0-1) to bars (0-5)
  const bars = Math.round(confidence * 5);
  return '●'.repeat(bars) + '○'.repeat(5 - bars);
}

/**
 * Format probability for different display contexts
 * @param {number} probability - Raw probability value (0-1)
 * @param {string} format - Format type ('percent', 'decimal', 'odds', 'fraction')
 * @returns {string} - Formatted probability
 */
function formatProbability(probability, format = 'percent') {
  switch (format.toLowerCase()) {
    case 'percent':
      return `${(probability * 100).toFixed(1)}%`;
      
    case 'decimal':
      return probability.toFixed(2);
      
    case 'odds':
      return `${(1 / probability).toFixed(2)}`;
      
    case 'fraction':
      return toFraction(probability);
      
    default:
      return `${(probability * 100).toFixed(1)}%`;
  }
}

/**
 * Convert decimal to approximate fraction representation
 * @private
 */
function toFraction(decimal) {
  if (decimal === 0) return '0/1';
  if (decimal === 1) return '1/1';
  
  const tolerance = 1.0E-6;
  
  // Find a simple fraction approximation
  let h1=1, h2=0, k1=0, k2=1;
  let b = decimal;
  
  do {
    const a = Math.floor(b);
    let aux = h1;
    h1 = a * h1 + h2;
    h2 = aux;
    aux = k1;
    k1 = a * k1 + k2;
    k2 = aux;
    b = 1 / (b - a);
  } while (Math.abs(decimal - h1/k1) > decimal * tolerance);
  
  return `${h1}/${k1}`;
}

module.exports = {
  transformMarketForFrontend,
  transformForecastForFrontend,
  transformComparisonForFrontend,
  transformEvidenceForFrontend,
  transformHistoricalDataForCharts,
  formatProbability
};