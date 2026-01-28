/**
 * Market-Model Comparison
 * 
 * This module analyzes differences between market-implied probabilities
 * and model-generated probabilities to identify opportunities.
 */

const math = require('mathjs');

class MarketModelComparison {
  constructor(config = {}) {
    this.config = {
      // Thresholds for divergence significance
      minorDivergenceThreshold: 0.05, // 5% difference
      majorDivergenceThreshold: 0.15, // 15% difference
      
      // Confidence adjustment factors
      marketLiquidityWeight: 0.3,
      modelConfidenceWeight: 0.5,
      historicalAccuracyWeight: 0.2,
      
      // Time decay parameters for historical performance
      timeDecayFactor: 0.9,
      
      ...config
    };
  }

  /**
   * Calculate the key metrics comparing market and model probabilities
   * @param {Object} marketData - Market-implied probability data
   * @param {Object} modelData - Model-generated probability data
   * @returns {Object} - Comparison metrics
   */
  calculateComparisonMetrics(marketData, modelData) {
    // Calculate absolute divergence
    const absoluteDivergence = Math.abs(marketData.probability - modelData.probability);
    
    // Calculate relative divergence (as percentage of market probability)
    const relativeDivergence = marketData.probability !== 0 ? 
      absoluteDivergence / marketData.probability : 
      absoluteDivergence;
    
    // Calculate direction of divergence (positive if model > market)
    const divergenceDirection = Math.sign(modelData.probability - marketData.probability);
    
    // Calculate Jensen-Shannon divergence for probability distributions
    // (if full distributions are available)
    let jsDiv = null;
    if (marketData.distribution && modelData.distribution) {
      jsDiv = this.calculateJSdivergence(marketData.distribution, modelData.distribution);
    }
    
    // Determine significance level
    let divergenceSignificance;
    if (absoluteDivergence < this.config.minorDivergenceThreshold) {
      divergenceSignificance = 'negligible';
    } else if (absoluteDivergence < this.config.majorDivergenceThreshold) {
      divergenceSignificance = 'minor';
    } else {
      divergenceSignificance = 'major';
    }
    
    return {
      absoluteDivergence,
      relativeDivergence,
      divergenceDirection,
      jsDiv,
      divergenceSignificance
    };
  }

  /**
   * Calculate Jensen-Shannon divergence between two probability distributions
   * @private
   */
  calculateJSdivergence(dist1, dist2) {
    // Ensure distributions are normalized and have the same length
    const len = Math.max(dist1.length, dist2.length);
    const p = new Array(len).fill(0);
    const q = new Array(len).fill(0);
    
    // Fill arrays with normalized distributions
    for (let i = 0; i < dist1.length; i++) {
      p[i] = dist1[i];
    }
    
    for (let i = 0; i < dist2.length; i++) {
      q[i] = dist2[i];
    }
    
    // Normalize
    const sumP = p.reduce((a, b) => a + b, 0);
    const sumQ = q.reduce((a, b) => a + b, 0);
    
    for (let i = 0; i < len; i++) {
      p[i] = p[i] / sumP;
      q[i] = q[i] / sumQ;
    }
    
    // Calculate midpoint distribution
    const m = p.map((val, idx) => (val + q[idx]) / 2);
    
    // Calculate KL divergence from P to M and Q to M
    let klPM = 0;
    let klQM = 0;
    
    for (let i = 0; i < len; i++) {
      if (p[i] > 0 && m[i] > 0) {
        klPM += p[i] * Math.log2(p[i] / m[i]);
      }
      
      if (q[i] > 0 && m[i] > 0) {
        klQM += q[i] * Math.log2(q[i] / m[i]);
      }
    }
    
    // JS divergence is the average of the two KL divergences
    return (klPM + klQM) / 2;
  }

  /**
   * Calculate the expected value of acting on the divergence
   * @param {Object} marketData - Market probability data
   * @param {Object} modelData - Model probability data
   * @param {Object} confidenceParams - Parameters for confidence adjustment
   * @returns {Object} - Edge calculation
   */
  calculateEdge(marketData, modelData, confidenceParams = {}) {
    // Calculate raw edge (how much model thinks market is mispriced)
    const rawEdge = modelData.probability - marketData.probability;
    
    // Adjust for confidence factors
    const marketLiquidity = confidenceParams.marketLiquidity || 0.5;
    const modelConfidence = confidenceParams.modelConfidence || 0.5;
    const historicalAccuracy = confidenceParams.historicalAccuracy || 0.5;
    
    // Weighted confidence score (0-1)
    const confidenceScore = 
      (marketLiquidity * this.config.marketLiquidityWeight) +
      (modelConfidence * this.config.modelConfidenceWeight) +
      (historicalAccuracy * this.config.historicalAccuracyWeight);
    
    // Adjusted edge - raw edge scaled by confidence
    const adjustedEdge = rawEdge * confidenceScore;
    
    // Calculate Kelly criterion for optimal position sizing
    // f* = p - (1-p)/(b) where b is the odds received (price/payout ratio)
    const kellyFraction = (() => {
      if (rawEdge <= 0) return 0; // No edge, no position
      
      const odds = (1 - marketData.probability) / marketData.probability;
      const p = modelData.probability;
      const kf = p - (1-p)/odds;
      
      // Cap kelly for risk management
      return Math.max(0, Math.min(0.5, kf));
    })();
    
    return {
      rawEdge,
      confidenceScore,
      adjustedEdge,
      kellyFraction,
      recommendation: this.generateRecommendation(adjustedEdge, confidenceScore)
    };
  }

  /**
   * Generate trading recommendation based on edge and confidence
   * @private
   */
  generateRecommendation(edge, confidence) {
    const edgeAbs = Math.abs(edge);
    const edgeSign = Math.sign(edge);
    
    // No significant edge
    if (edgeAbs < 0.02) {
      return {
        action: 'HOLD',
        strength: 'weak',
        reasoning: 'No significant difference between market and model probabilities'
      };
    }
    
    // Low confidence
    if (confidence < 0.4) {
      return {
        action: edgeSign > 0 ? 'BUY' : 'SELL',
        strength: 'weak',
        reasoning: 'Some edge detected but confidence is low, proceed with caution'
      };
    }
    
    // Medium edge
    if (edgeAbs < 0.1) {
      return {
        action: edgeSign > 0 ? 'BUY' : 'SELL',
        strength: 'moderate',
        reasoning: 'Moderate edge with sufficient confidence'
      };
    }
    
    // Strong edge with good confidence
    return {
      action: edgeSign > 0 ? 'BUY' : 'SELL',
      strength: 'strong',
      reasoning: 'Strong edge with good confidence, significant mispricing detected'
    };
  }

  /**
   * Performs a comprehensive comparison between market and model data
   * @param {Object} marketData - Market probability data
   * @param {Object} modelData - Model probability data
   * @param {Object} additionalParams - Additional parameters
   * @returns {Object} - Comprehensive comparison
   */
  compareMarketToModel(marketData, modelData, additionalParams = {}) {
    // Basic validation
    if (!marketData || !modelData) {
      throw new Error('Market and model data are required');
    }
    
    if (typeof marketData.probability !== 'number' || typeof modelData.probability !== 'number') {
      throw new Error('Probability values must be numbers');
    }
    
    // Calculate metrics
    const metrics = this.calculateComparisonMetrics(marketData, modelData);
    
    // Calculate edge
    const edge = this.calculateEdge(marketData, modelData, {
      marketLiquidity: additionalParams.marketLiquidity || 0.5,
      modelConfidence: additionalParams.modelConfidence || 0.5,
      historicalAccuracy: additionalParams.historicalAccuracy || 0.5,
    });
    
    // Generate report
    return {
      timestamp: new Date().toISOString(),
      market: {
        probability: marketData.probability,
        impliedOdds: (1 / marketData.probability).toFixed(2),
        source: marketData.source || 'unknown',
        timestamp: marketData.timestamp
      },
      model: {
        probability: modelData.probability,
        confidence: modelData.confidence || 'unknown',
        methodology: modelData.methodology || 'unknown',
        timestamp: modelData.timestamp
      },
      comparison: {
        ...metrics,
        edge,
        opportunity: metrics.divergenceSignificance !== 'negligible',
        summary: `The model ${edge.recommendation.action === 'BUY' ? 'favors' : 'opposes'} this outcome with ${edge.recommendation.strength} conviction, showing a ${(metrics.absoluteDivergence * 100).toFixed(1)}% divergence from market price.`
      }
    };
  }
}

module.exports = MarketModelComparison;