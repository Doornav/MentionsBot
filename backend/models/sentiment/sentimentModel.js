/**
 * Sentiment Analysis Model
 * 
 * This model analyzes text data to extract sentiment and relevance
 * to help inform predictions and provide evidence.
 */

const natural = require('natural');
const tokenizer = new natural.WordTokenizer();
const { TfIdf } = natural;

class SentimentModel {
  constructor(config = {}) {
    this.config = {
      negationDistanceThreshold: 5, // How close negation words affect sentiment
      minWordLength: 2,            // Ignore very short words
      useNegation: true,           // Consider negations like "not good"
      ...config
    };
    
    // Initialize AFINN sentiment lexicon
    this.sentiment = new natural.SentimentAnalyzer(
      'English', 
      natural.PorterStemmer, 
      'afinn'
    );
    
    // Initialize TF-IDF for relevance scoring
    this.tfidf = new TfIdf();
    
    // Domain-specific terms to boost
    this.domainTerms = {
      'prediction': 1.5,
      'forecast': 1.5,
      'market': 1.2,
      'probability': 1.3,
      'trend': 1.2,
      'risk': 1.3,
      'evidence': 1.4,
      'data': 1.2,
      'analyst': 1.2,
      'indicator': 1.3,
    };
  }

  /**
   * Preprocess text before analysis
   * @param {string} text - Raw text to process
   * @returns {Array} - Array of processed tokens
   */
  preprocess(text) {
    // Convert to lowercase
    const lowerText = text.toLowerCase();
    
    // Tokenize
    const tokens = tokenizer.tokenize(lowerText);
    
    // Filter tokens
    return tokens.filter(token => 
      token.length >= this.config.minWordLength &&
      !natural.stopwords.includes(token)
    );
  }

  /**
   * Analyze the sentiment of a text
   * @param {string} text - Text to analyze
   * @returns {Object} - Sentiment analysis results
   */
  analyzeSentiment(text) {
    const tokens = this.preprocess(text);
    
    // Basic sentiment score
    let sentimentScore = this.sentiment.getSentiment(tokens);
    
    // Handle negations if enabled
    if (this.config.useNegation) {
      const negationWords = ['not', 'no', 'never', "don't", "doesn't", 'cannot', "can't"];
      
      for (let i = 0; i < tokens.length; i++) {
        if (negationWords.includes(tokens[i])) {
          // Look ahead up to the negation threshold and invert sentiment words
          for (let j = i + 1; j < Math.min(i + this.config.negationDistanceThreshold, tokens.length); j++) {
            const wordScore = this.sentiment.getSentiment([tokens[j]]);
            if (Math.abs(wordScore) > 0.1) {
              // Adjust the overall sentiment score by negating this word's contribution
              sentimentScore -= 2 * wordScore; // Double negation for emphasis
              break; // Only negate the first sentiment word found
            }
          }
        }
      }
    }
    
    // Categorize the sentiment
    let sentimentCategory;
    if (sentimentScore < -0.05) sentimentCategory = 'negative';
    else if (sentimentScore > 0.05) sentimentCategory = 'positive';
    else sentimentCategory = 'neutral';
    
    // Calculate sentiment intensity (0-1)
    const sentimentIntensity = Math.min(Math.abs(sentimentScore) * 2, 1);
    
    return {
      score: sentimentScore,
      category: sentimentCategory,
      intensity: sentimentIntensity,
      tokens: tokens.length
    };
  }

  /**
   * Calculate the relevance of a text to a specific topic
   * @param {string} text - Text to analyze
   * @param {string} topic - Topic to compare against
   * @param {Object} topicKeywords - Additional keywords related to the topic
   * @returns {Object} - Relevance analysis results
   */
  calculateRelevance(text, topic, topicKeywords = {}) {
    // Preprocess the text
    const tokens = this.preprocess(text);
    const topicTokens = this.preprocess(topic);
    
    // Calculate basic term overlap
    const textSet = new Set(tokens);
    const topicSet = new Set(topicTokens);
    const intersection = new Set([...textSet].filter(x => topicSet.has(x)));
    
    const overlapScore = intersection.size / topicSet.size;
    
    // Use TF-IDF for more sophisticated relevance
    this.tfidf = new TfIdf();
    this.tfidf.addDocument(topicTokens.join(' '));
    this.tfidf.addDocument(tokens.join(' '));
    
    let tfidfScore = 0;
    this.tfidf.tfidfs(topicTokens.join(' '), (i, measure) => {
      if (i === 1) { // The document we're analyzing
        tfidfScore = measure;
      }
    });
    
    // Boost score based on domain-specific terms
    let domainBoost = 1.0;
    for (const token of tokens) {
      if (this.domainTerms[token]) {
        domainBoost += 0.05 * this.domainTerms[token];
      }
    }
    
    // Include custom topic keywords
    let keywordBoost = 0;
    for (const token of tokens) {
      if (topicKeywords[token]) {
        keywordBoost += 0.1 * topicKeywords[token];
      }
    }
    
    // Calculate final relevance score (normalized 0-1)
    const rawRelevance = (overlapScore * 0.3 + tfidfScore * 0.4) * domainBoost + keywordBoost;
    const relevanceScore = Math.min(Math.max(rawRelevance, 0), 1);
    
    // Categorize relevance
    let relevanceCategory;
    if (relevanceScore < 0.3) relevanceCategory = 'low';
    else if (relevanceScore < 0.7) relevanceCategory = 'medium';
    else relevanceCategory = 'high';
    
    return {
      score: relevanceScore,
      category: relevanceCategory,
      matchedTerms: Array.from(intersection),
      domainBoost: domainBoost
    };
  }

  /**
   * Analyze text to extract both sentiment and relevance
   * @param {string} text - Text to analyze
   * @param {string} topic - Topic for relevance comparison
   * @param {Object} options - Additional analysis options
   * @returns {Object} - Complete analysis results
   */
  analyzeText(text, topic, options = {}) {
    const sentiment = this.analyzeSentiment(text);
    const relevance = this.calculateRelevance(text, topic, options.topicKeywords);
    
    // Calculate confidence based on text length, relevance and sentiment intensity
    const confidenceFactors = {
      textLength: Math.min(text.length / 1000, 1) * 0.3,
      relevance: relevance.score * 0.5,
      sentimentClarity: sentiment.intensity * 0.2
    };
    
    const confidenceScore = Object.values(confidenceFactors).reduce((sum, factor) => sum + factor, 0);
    
    // Extract key phrases (simple for now - could be enhanced with NLP libraries)
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
    const keyPhrases = sentences
      .filter(s => {
        const sentimentScore = this.analyzeSentiment(s).score;
        return Math.abs(sentimentScore) > 0.2; // Only strong sentiment sentences
      })
      .slice(0, 3);
    
    return {
      sentiment,
      relevance,
      confidence: confidenceScore,
      confidenceFactors,
      keyPhrases,
      timestamp: new Date().toISOString(),
      metadata: {
        textLength: text.length,
        sentences: sentences.length,
      }
    };
  }

  /**
   * Train the model with additional domain-specific data
   * @param {Array} trainingData - Array of labeled examples
   */
  train(trainingData) {
    // Extend domain terms based on training data
    for (const example of trainingData) {
      const tokens = this.preprocess(example.text);
      
      if (example.relevance > 0.7) {
        // For highly relevant examples, boost terms
        for (const token of tokens) {
          if (!this.domainTerms[token]) {
            this.domainTerms[token] = 1.0;
          } else {
            this.domainTerms[token] += 0.1;
          }
        }
      }
    }
    
    console.log(`Updated domain terms dictionary with ${Object.keys(this.domainTerms).length} entries`);
    return true;
  }
}

module.exports = SentimentModel;