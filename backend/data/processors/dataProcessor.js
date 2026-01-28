/**
 * Data Processor
 * 
 * General purpose data processing utility for handling
 * raw data from different sources and transforming it
 * into standardized formats for model consumption.
 */

class DataProcessor {
  constructor(config = {}) {
    this.config = {
      defaultDateFormat: 'ISO',
      probabilityPrecision: 4,
      ...config
    };
  }

  /**
   * Process raw market data into standardized format
   * @param {Object} rawData - Raw data from source
   * @param {string} source - Source identifier
   * @returns {Object} - Processed data
   */
  processMarketData(rawData, source) {
    // Ensure we have a valid data object
    if (!rawData) {
      throw new Error('Raw data is missing or null');
    }
    
    // Apply source-specific transformations
    const processed = { ...rawData };
    
    // Standardize probability format (ensure it's between 0 and 1)
    if (typeof processed.probability !== 'undefined') {
      processed.probability = this.standardizeProbability(processed.probability, source);
    }
    
    // Standardize dates
    if (processed.closeTime) {
      processed.closeTime = this.standardizeDate(processed.closeTime);
    }
    
    // Add additional calculated fields
    processed.impliedOdds = processed.probability ? (1 / processed.probability) : null;
    
    // Add processing metadata
    processed.processedAt = new Date().toISOString();
    processed.processingVersion = '1.0.0';
    
    return processed;
  }

  /**
   * Standardize probability value
   * @private
   */
  standardizeProbability(value, source) {
    // Handle different source formats
    let probability = value;
    
    switch (source) {
      case 'metaculus':
        // Metaculus sometimes provides probabilities as percentages
        if (probability > 1) {
          probability = probability / 100;
        }
        break;
        
      case 'predictit':
        // PredictIt sometimes represents as cents
        if (probability > 1) {
          probability = probability / 100;
        }
        break;
      
      // Add more sources as needed
    }
    
    // Enforce bounds
    probability = Math.max(0, Math.min(1, probability));
    
    // Round to configured precision
    return Number(probability.toFixed(this.config.probabilityPrecision));
  }

  /**
   * Standardize date format
   * @private
   */
  standardizeDate(dateValue) {
    // If it's already an ISO string, return it
    if (typeof dateValue === 'string' && dateValue.match(/^\d{4}-\d{2}-\d{2}T/)) {
      return dateValue;
    }
    
    // If it's a timestamp in milliseconds
    if (typeof dateValue === 'number') {
      return new Date(dateValue).toISOString();
    }
    
    // Otherwise try to parse as Date
    try {
      return new Date(dateValue).toISOString();
    } catch (e) {
      console.error('Failed to parse date:', dateValue);
      return null;
    }
  }

  /**
   * Process time series data
   * @param {Array} dataPoints - Array of data points
   * @param {Object} options - Processing options
   * @returns {Object} - Processed time series
   */
  processTimeSeries(dataPoints, options = {}) {
    const defaultOptions = {
      fillMissing: true,
      smoothing: false,
      smoothingWindow: 3,
      normalization: false
    };
    
    const opts = { ...defaultOptions, ...options };
    
    // Ensure data points are sorted by timestamp
    const sortedPoints = [...dataPoints].sort((a, b) => {
      const dateA = new Date(a.timestamp);
      const dateB = new Date(b.timestamp);
      return dateA - dateB;
    });
    
    // Extract the probability values
    let probabilities = sortedPoints.map(p => p.probability);
    
    // Fill missing values if enabled
    if (opts.fillMissing) {
      probabilities = this.fillMissingValues(probabilities);
    }
    
    // Apply smoothing if enabled
    if (opts.smoothing) {
      probabilities = this.smoothTimeSeries(probabilities, opts.smoothingWindow);
    }
    
    // Apply normalization if enabled
    if (opts.normalization) {
      probabilities = this.normalizeValues(probabilities);
    }
    
    // Reconstruct the time series
    const processedSeries = sortedPoints.map((point, index) => ({
      ...point,
      probability: probabilities[index],
      processed: true
    }));
    
    return {
      dataPoints: processedSeries,
      metadata: {
        originalLength: dataPoints.length,
        processedLength: processedSeries.length,
        startDate: processedSeries[0].timestamp,
        endDate: processedSeries[processedSeries.length - 1].timestamp,
        options: opts
      }
    };
  }

  /**
   * Fill missing values in a time series
   * @private
   */
  fillMissingValues(values) {
    const result = [...values];
    
    // Find missing values (null or undefined)
    for (let i = 0; i < result.length; i++) {
      if (result[i] === null || result[i] === undefined) {
        // Find the nearest valid values before and after
        let before = null;
        let after = null;
        
        // Look backward for valid value
        for (let j = i - 1; j >= 0; j--) {
          if (result[j] !== null && result[j] !== undefined) {
            before = { index: j, value: result[j] };
            break;
          }
        }
        
        // Look forward for valid value
        for (let j = i + 1; j < result.length; j++) {
          if (result[j] !== null && result[j] !== undefined) {
            after = { index: j, value: result[j] };
            break;
          }
        }
        
        // Fill with appropriate value
        if (before && after) {
          // Linear interpolation
          const position = (i - before.index) / (after.index - before.index);
          result[i] = before.value + position * (after.value - before.value);
        } else if (before) {
          // If only have value before, use that
          result[i] = before.value;
        } else if (after) {
          // If only have value after, use that
          result[i] = after.value;
        } else {
          // No valid values found, use default
          result[i] = 0.5; // Middle probability
        }
      }
    }
    
    return result;
  }

  /**
   * Smooth a time series using moving average
   * @private
   */
  smoothTimeSeries(values, windowSize = 3) {
    const result = [...values];
    
    // Need enough data points for smoothing
    if (values.length < windowSize) {
      return result;
    }
    
    // Apply moving average
    for (let i = 0; i < result.length; i++) {
      let sum = 0;
      let count = 0;
      
      // Gather values within window
      for (let j = Math.max(0, i - Math.floor(windowSize / 2)); 
           j <= Math.min(result.length - 1, i + Math.floor(windowSize / 2)); 
           j++) {
        if (values[j] !== null && values[j] !== undefined) {
          sum += values[j];
          count++;
        }
      }
      
      // Calculate average if we have valid values
      if (count > 0) {
        result[i] = sum / count;
      }
    }
    
    return result;
  }

  /**
   * Normalize values to [0,1] range
   * @private
   */
  normalizeValues(values) {
    // Find min and max
    const validValues = values.filter(v => v !== null && v !== undefined);
    
    if (validValues.length === 0) {
      return values;
    }
    
    const min = Math.min(...validValues);
    const max = Math.max(...validValues);
    
    // If all values are the same, return as is
    if (min === max) {
      return values;
    }
    
    // Normalize
    return values.map(v => {
      if (v === null || v === undefined) {
        return v;
      }
      return (v - min) / (max - min);
    });
  }

  /**
   * Process and merge evidence data
   * @param {Array} evidenceItems - Raw evidence items
   * @param {Object} options - Processing options
   * @returns {Object} - Processed evidence with aggregated impact
   */
  processEvidenceData(evidenceItems, options = {}) {
    const defaultOptions = {
      weightByRelevance: true,
      weightByRecency: true,
      recencyHalfLife: 7, // days
      credibilityThreshold: 0.3
    };
    
    const opts = { ...defaultOptions, ...options };
    
    // Filter low-credibility items
    const credibleItems = evidenceItems.filter(item => 
      item.confidence >= opts.credibilityThreshold
    );
    
    // Calculate weights for each item
    const weightedItems = credibleItems.map(item => {
      // Base weight
      let weight = 1.0;
      
      // Weight by relevance if enabled
      if (opts.weightByRelevance && typeof item.relevance === 'number') {
        weight *= item.relevance;
      }
      
      // Weight by recency if enabled
      if (opts.weightByRecency && item.timestamp) {
        const ageInDays = (Date.now() - new Date(item.timestamp).getTime()) / (1000 * 60 * 60 * 24);
        weight *= Math.exp(-Math.log(2) * ageInDays / opts.recencyHalfLife);
      }
      
      return {
        ...item,
        weight
      };
    });
    
    // Calculate aggregated sentiment
    const totalWeight = weightedItems.reduce((sum, item) => sum + item.weight, 0);
    
    if (totalWeight === 0) {
      return {
        items: weightedItems,
        aggregatedImpact: 0,
        weightedItemCount: 0
      };
    }
    
    const aggregatedSentiment = weightedItems.reduce(
      (sum, item) => sum + (item.sentiment * item.weight), 
      0
    ) / totalWeight;
    
    // Map sentiment to probability impact (-1 to 1 scale → probability shift)
    const aggregatedImpact = aggregatedSentiment * 0.1; // Max 10% shift from evidence
    
    return {
      items: weightedItems,
      aggregatedSentiment,
      aggregatedImpact,
      weightedItemCount: weightedItems.length,
      totalWeight
    };
  }
}

module.exports = DataProcessor;