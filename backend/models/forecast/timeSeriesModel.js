/**
 * Time Series Forecasting Model
 * 
 * This model uses historical data to forecast future probabilities
 * using techniques like ARIMA, Exponential Smoothing, or RNNs.
 */

const tf = require('@tensorflow/tfjs-node');
const math = require('mathjs');

class TimeSeriesModel {
  constructor(config = {}) {
    this.config = {
      sequenceLength: 14,    // Number of days to use for sequence input
      forecastHorizon: 7,    // Number of days to forecast ahead
      hiddenLayers: [64, 32], // Architecture of RNN
      learningRate: 0.001,
      ...config
    };
    
    this.model = null;
  }

  /**
   * Preprocess time series data
   * @param {Array} data - Array of time series values
   * @returns {Object} - Processed data for model input
   */
  preprocessData(data) {
    // Normalize the data
    const min = Math.min(...data);
    const max = Math.max(...data);
    const normalizedData = data.map(x => (x - min) / (max - min));
    
    // Create sequences for training
    const sequences = [];
    const targets = [];
    
    for (let i = 0; i <= normalizedData.length - this.config.sequenceLength - this.config.forecastHorizon; i++) {
      const sequence = normalizedData.slice(i, i + this.config.sequenceLength);
      const target = normalizedData[i + this.config.sequenceLength + this.config.forecastHorizon - 1];
      sequences.push(sequence);
      targets.push(target);
    }
    
    return {
      sequences,
      targets,
      min,
      max
    };
  }

  /**
   * Build the LSTM model architecture
   * @private
   */
  buildModel() {
    const model = tf.sequential();
    
    // Input layer
    model.add(tf.layers.lstm({
      units: this.config.hiddenLayers[0],
      inputShape: [this.config.sequenceLength, 1],
      returnSequences: this.config.hiddenLayers.length > 1
    }));
    
    // Hidden layers
    for (let i = 1; i < this.config.hiddenLayers.length; i++) {
      model.add(tf.layers.lstm({
        units: this.config.hiddenLayers[i],
        returnSequences: i < this.config.hiddenLayers.length - 1
      }));
    }
    
    // Output layer
    model.add(tf.layers.dense({ units: 1 }));
    
    // Compile the model
    model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: 'meanSquaredError'
    });
    
    this.model = model;
    return model;
  }

  /**
   * Train the model with historical data
   * @param {Array} data - Historical time series data
   * @param {Object} trainingConfig - Training configuration
   * @returns {Promise} - Training history
   */
  async train(data, trainingConfig = {}) {
    const defaultConfig = {
      epochs: 50,
      batchSize: 32,
      validationSplit: 0.2
    };
    
    const config = { ...defaultConfig, ...trainingConfig };
    const processedData = this.preprocessData(data);
    
    // Create tensors
    const xTensor = tf.tensor3d(
      processedData.sequences.map(seq => seq.map(val => [val])) 
    );
    
    const yTensor = tf.tensor2d(
      processedData.targets.map(val => [val])
    );
    
    // Build the model if not already built
    if (!this.model) {
      this.buildModel();
    }
    
    // Train the model
    const history = await this.model.fit(xTensor, yTensor, {
      epochs: config.epochs,
      batchSize: config.batchSize,
      validationSplit: config.validationSplit,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch}: loss = ${logs.loss}, val_loss = ${logs.val_loss}`);
        }
      }
    });
    
    // Clean up tensors
    xTensor.dispose();
    yTensor.dispose();
    
    // Save metadata for denormalization
    this.metadata = {
      min: processedData.min,
      max: processedData.max
    };
    
    return history;
  }

  /**
   * Make a forecast based on recent data
   * @param {Array} recentData - Recent time series data
   * @returns {Array} - Forecasted values
   */
  forecast(recentData) {
    // Ensure model is built
    if (!this.model) {
      throw new Error('Model not trained. Call train() first.');
    }
    
    // Ensure we have enough data
    if (recentData.length < this.config.sequenceLength) {
      throw new Error(`Need at least ${this.config.sequenceLength} data points, got ${recentData.length}`);
    }
    
    // Normalize data using saved metadata
    const normalizedData = recentData.map(x => 
      (x - this.metadata.min) / (this.metadata.max - this.metadata.min)
    );
    
    // Use the most recent window of data
    const inputSequence = normalizedData.slice(-this.config.sequenceLength);
    const inputTensor = tf.tensor3d([inputSequence.map(val => [val])]);
    
    // Make prediction
    const forecastTensor = this.model.predict(inputTensor);
    const forecastData = forecastTensor.dataSync();
    
    // Denormalize the output
    const forecast = forecastData.map(val => 
      val * (this.metadata.max - this.metadata.min) + this.metadata.min
    );
    
    // Clean up tensors
    inputTensor.dispose();
    forecastTensor.dispose();
    
    return forecast;
  }

  /**
   * Save the model to a file
   * @param {string} path - Path to save the model
   */
  async saveModel(path) {
    if (!this.model) {
      throw new Error('No model to save. Train first.');
    }
    
    await this.model.save(`file://${path}`);
    
    // Save metadata separately
    const fs = require('fs');
    fs.writeFileSync(`${path}_metadata.json`, JSON.stringify(this.metadata));
    
    return true;
  }

  /**
   * Load a model from a file
   * @param {string} path - Path to load the model from
   */
  async loadModel(path) {
    this.model = await tf.loadLayersModel(`file://${path}`);
    
    // Load metadata
    const fs = require('fs');
    this.metadata = JSON.parse(fs.readFileSync(`${path}_metadata.json`));
    
    return this.model;
  }
}

module.exports = TimeSeriesModel;