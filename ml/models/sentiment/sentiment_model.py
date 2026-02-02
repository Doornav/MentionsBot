"""
Sentiment Analysis Model

This model analyzes text data to extract sentiment and relevance
to help inform predictions and provide evidence.
"""

import re
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Tuple, Union, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentModel:
    def __init__(self, config: Dict = None):
        """
        Initialize the sentiment analysis model
        
        Args:
            config: Configuration dictionary for the model
        """
        self.config = {
            "negation_distance_threshold": 5,  # How close negation words affect sentiment
            "min_word_length": 2,             # Ignore very short words
            "use_negation": True,             # Consider negations like "not good"
            "use_transformer": True,          # Use transformer model for sentiment
            "transformer_model": "distilbert-base-uncased-finetuned-sst-2-english",
            "batch_size": 8,
            **(config or {})
        }
        
        # Download necessary NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Set up stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Set up TF-IDF for relevance scoring
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.95, stop_words='english'
        )
        
        # Domain-specific terms to boost
        self.domain_terms = {
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
        }
        
        # Initialize transformer model for sentiment analysis if enabled
        if self.config["use_transformer"]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["transformer_model"])
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config["transformer_model"])
            
            # Check if GPU is available and move model to it
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text before analysis
        
        Args:
            text: Raw text to process
            
        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        lower_text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(lower_text)
        
        # Filter tokens
        return [
            token for token in tokens 
            if len(token) >= self.config["min_word_length"] 
            and token not in self.stop_words
        ]
    
    def analyze_sentiment_transformer(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer model
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Prepare inputs
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.cpu().numpy()[0]
        
        # Map to sentiment score (-1 to 1 range)
        # Assuming index 1 is positive sentiment, 0 is negative
        sentiment_score = (probs[1] - probs[0])
        
        # Categorize sentiment
        if sentiment_score < -0.05:
            sentiment_category = 'negative'
        elif sentiment_score > 0.05:
            sentiment_category = 'positive'
        else:
            sentiment_category = 'neutral'
        
        # Calculate sentiment intensity (0-1)
        sentiment_intensity = min(abs(sentiment_score) * 2, 1)
        
        return {
            "score": sentiment_score,
            "category": sentiment_category,
            "intensity": sentiment_intensity,
            "raw_probabilities": {
                "negative": float(probs[0]),
                "positive": float(probs[1])
            }
        }
    
    def analyze_sentiment_lexicon(self, text: str) -> Dict:
        """
        Analyze sentiment using a lexicon-based approach
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        tokens = self.preprocess(text)
        
        # Simplified VADER-style sentiment calculation
        # This would be replaced with NLTK's SentimentIntensityAnalyzer in production
        
        # Simplified sentiment dictionary for demonstration
        pos_words = {
            "good": 0.5, "great": 0.8, "excellent": 1.0, "positive": 0.6,
            "rise": 0.4, "increase": 0.4, "growth": 0.5, "profit": 0.7,
            "up": 0.3, "higher": 0.4, "best": 0.9, "gain": 0.6
        }
        
        neg_words = {
            "bad": -0.5, "terrible": -0.8, "awful": -1.0, "negative": -0.6,
            "fall": -0.4, "decrease": -0.4, "decline": -0.5, "loss": -0.7,
            "down": -0.3, "lower": -0.4, "worst": -0.9, "lose": -0.6
        }
        
        # Calculate base sentiment score
        sentiment_score = 0
        for token in tokens:
            if token in pos_words:
                sentiment_score += pos_words[token]
            elif token in neg_words:
                sentiment_score += neg_words[token]
        
        # Normalize by token count
        if tokens:
            sentiment_score = sentiment_score / len(tokens) * 3  # Scale to be comparable to transformer
        
        # Handle negations if enabled
        if self.config["use_negation"]:
            negation_words = ['not', 'no', 'never', "don't", "doesn't", 'cannot', "can't"]
            
            for i, token in enumerate(tokens):
                if token in negation_words:
                    # Look ahead up to the negation threshold and invert sentiment words
                    for j in range(i + 1, min(i + self.config["negation_distance_threshold"], len(tokens))):
                        current_token = tokens[j]
                        word_score = 0
                        
                        if current_token in pos_words:
                            word_score = pos_words[current_token]
                        elif current_token in neg_words:
                            word_score = neg_words[current_token]
                            
                        if abs(word_score) > 0.1:
                            # Adjust the overall sentiment score by negating this word's contribution
                            sentiment_score -= 2 * word_score  # Double negation for emphasis
                            break  # Only negate the first sentiment word found
        
        # Categorize sentiment
        if sentiment_score < -0.05:
            sentiment_category = 'negative'
        elif sentiment_score > 0.05:
            sentiment_category = 'positive'
        else:
            sentiment_category = 'neutral'
        
        # Calculate sentiment intensity (0-1)
        sentiment_intensity = min(abs(sentiment_score) * 2, 1)
        
        return {
            "score": sentiment_score,
            "category": sentiment_category,
            "intensity": sentiment_intensity,
            "tokens": len(tokens)
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Main method to analyze sentiment, choosing between transformer and lexicon-based
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if self.config["use_transformer"]:
            return self.analyze_sentiment_transformer(text)
        else:
            return self.analyze_sentiment_lexicon(text)
    
    def calculate_relevance(self, text: str, topic: str, topic_keywords: Dict = None) -> Dict:
        """
        Calculate the relevance of a text to a specific topic
        
        Args:
            text: Text to analyze
            topic: Topic to compare against
            topic_keywords: Additional keywords related to the topic
            
        Returns:
            Relevance analysis results
        """
        topic_keywords = topic_keywords or {}
        
        # Preprocess the text
        tokens = self.preprocess(text)
        topic_tokens = self.preprocess(topic)
        
        # Calculate basic term overlap
        text_set = set(tokens)
        topic_set = set(topic_tokens)
        intersection = text_set.intersection(topic_set)
        
        overlap_score = len(intersection) / len(topic_set) if topic_set else 0
        
        # Use TF-IDF for more sophisticated relevance
        try:
            # Create a corpus with both texts
            corpus = [topic, text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity between topic and text
            similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
            tfidf_score = similarity
        except:
            # Fallback if TF-IDF fails (e.g., too few documents)
            tfidf_score = overlap_score
        
        # Boost score based on domain-specific terms
        domain_boost = 1.0
        for token in tokens:
            if token in self.domain_terms:
                domain_boost += 0.05 * self.domain_terms[token]
        
        # Include custom topic keywords
        keyword_boost = 0
        for token in tokens:
            if token in topic_keywords:
                keyword_boost += 0.1 * topic_keywords[token]
        
        # Calculate final relevance score (normalized 0-1)
        raw_relevance = (overlap_score * 0.3 + tfidf_score * 0.4) * domain_boost + keyword_boost
        relevance_score = max(0, min(raw_relevance, 1))
        
        # Categorize relevance
        if relevance_score < 0.3:
            relevance_category = 'low'
        elif relevance_score < 0.7:
            relevance_category = 'medium'
        else:
            relevance_category = 'high'
        
        return {
            "score": relevance_score,
            "category": relevance_category,
            "matched_terms": list(intersection),
            "domain_boost": domain_boost
        }
    
    def analyze_text(self, text: str, topic: str, options: Dict = None) -> Dict:
        """
        Analyze text to extract both sentiment and relevance
        
        Args:
            text: Text to analyze
            topic: Topic for relevance comparison
            options: Additional analysis options
            
        Returns:
            Complete analysis results
        """
        options = options or {}
        
        sentiment = self.analyze_sentiment(text)
        relevance = self.calculate_relevance(text, topic, options.get("topic_keywords", {}))
        
        # Calculate confidence based on text length, relevance and sentiment intensity
        confidence_factors = {
            "text_length": min(len(text) / 1000, 1) * 0.3,
            "relevance": relevance["score"] * 0.5,
            "sentiment_clarity": sentiment.get("intensity", 0) * 0.2
        }
        
        confidence_score = sum(confidence_factors.values())
        
        # Extract key phrases (simple for now - could be enhanced with NLP libraries)
        sentences = re.findall(r"[^.!?]+[.!?]+", text) or []
        
        # Filter for strong sentiment sentences
        key_phrases = []
        for sentence in sentences:
            sentence_sentiment = self.analyze_sentiment(sentence)
            if abs(sentence_sentiment["score"]) > 0.2:
                key_phrases.append(sentence)
        
        key_phrases = key_phrases[:3]  # Take top 3
        
        return {
            "sentiment": sentiment,
            "relevance": relevance,
            "confidence": confidence_score,
            "confidence_factors": confidence_factors,
            "key_phrases": key_phrases,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "text_length": len(text),
                "sentences": len(sentences),
            }
        }
    
    def train(self, training_data: List[Dict]) -> bool:
        """
        Train the model with additional domain-specific data
        
        Args:
            training_data: Array of labeled examples
            
        Returns:
            True if training was successful
        """
        # Extend domain terms based on training data
        for example in training_data:
            if "text" not in example or "relevance" not in example:
                continue
                
            tokens = self.preprocess(example["text"])
            
            if example["relevance"] > 0.7:
                # For highly relevant examples, boost terms
                for token in tokens:
                    if token not in self.domain_terms:
                        self.domain_terms[token] = 1.0
                    else:
                        self.domain_terms[token] += 0.1
        
        print(f"Updated domain terms dictionary with {len(self.domain_terms)} entries")
        return True


if __name__ == "__main__":
    # Example usage
    model = SentimentModel()
    result = model.analyze_text(
        "The market is trending upward with strong economic indicators, suggesting a positive outlook.",
        "market trends",
        {"topic_keywords": {"economic": 1.5, "indicators": 1.3}}
    )
    print(result)