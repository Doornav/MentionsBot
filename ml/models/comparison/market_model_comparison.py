"""
Market-Model Comparison

This module analyzes differences between market-implied probabilities
and model-generated probabilities to identify opportunities.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime
import torch
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon


class MarketModelComparison:
    def __init__(self, config: Dict = None):
        """
        Initialize the comparison model
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            # Thresholds for divergence significance
            "minor_divergence_threshold": 0.05,  # 5% difference
            "major_divergence_threshold": 0.15,  # 15% difference
            
            # Confidence adjustment factors
            "market_liquidity_weight": 0.3,
            "model_confidence_weight": 0.5,
            "historical_accuracy_weight": 0.2,
            
            # Time decay parameters for historical performance
            "time_decay_factor": 0.9,
            
            **(config or {})
        }
    
    def calculate_comparison_metrics(self, market_data: Dict, model_data: Dict) -> Dict:
        """
        Calculate the key metrics comparing market and model probabilities
        
        Args:
            market_data: Market-implied probability data
            model_data: Model-generated probability data
            
        Returns:
            Comparison metrics
        """
        # Calculate absolute divergence
        absolute_divergence = abs(market_data["probability"] - model_data["probability"])
        
        # Calculate relative divergence (as percentage of market probability)
        relative_divergence = (
            absolute_divergence / market_data["probability"] 
            if market_data["probability"] != 0 
            else absolute_divergence
        )
        
        # Calculate direction of divergence (positive if model > market)
        divergence_direction = np.sign(model_data["probability"] - market_data["probability"])
        
        # Calculate Jensen-Shannon divergence for probability distributions
        # (if full distributions are available)
        js_div = None
        if "distribution" in market_data and "distribution" in model_data:
            js_div = self.calculate_js_divergence(
                market_data["distribution"], 
                model_data["distribution"]
            )
        
        # Determine significance level
        if absolute_divergence < self.config["minor_divergence_threshold"]:
            divergence_significance = 'negligible'
        elif absolute_divergence < self.config["major_divergence_threshold"]:
            divergence_significance = 'minor'
        else:
            divergence_significance = 'major'
        
        return {
            "absolute_divergence": absolute_divergence,
            "relative_divergence": relative_divergence,
            "divergence_direction": divergence_direction,
            "js_div": js_div,
            "divergence_significance": divergence_significance
        }
    
    def calculate_js_divergence(self, dist1: List[float], dist2: List[float]) -> float:
        """
        Calculate Jensen-Shannon divergence between two probability distributions
        
        Args:
            dist1: First probability distribution
            dist2: Second probability distribution
            
        Returns:
            JS divergence score
        """
        # Ensure distributions are numpy arrays
        dist1 = np.array(dist1)
        dist2 = np.array(dist2)
        
        # Ensure distributions are normalized and have the same length
        len_max = max(len(dist1), len(dist2))
        
        p = np.zeros(len_max)
        q = np.zeros(len_max)
        
        # Fill arrays with normalized distributions
        p[:len(dist1)] = dist1
        q[:len(dist2)] = dist2
        
        # Normalize
        p_sum = np.sum(p)
        q_sum = np.sum(q)
        
        if p_sum > 0:
            p = p / p_sum
        
        if q_sum > 0:
            q = q / q_sum
        
        # Use scipy's implementation of Jensen-Shannon divergence
        return jensenshannon(p, q)
    
    def calculate_edge(self, market_data: Dict, model_data: Dict, confidence_params: Dict = None) -> Dict:
        """
        Calculate the expected value of acting on the divergence
        
        Args:
            market_data: Market probability data
            model_data: Model probability data
            confidence_params: Parameters for confidence adjustment
            
        Returns:
            Edge calculation
        """
        confidence_params = confidence_params or {}
        
        # Calculate raw edge (how much model thinks market is mispriced)
        raw_edge = model_data["probability"] - market_data["probability"]
        
        # Adjust for confidence factors
        market_liquidity = confidence_params.get("market_liquidity", 0.5)
        model_confidence = confidence_params.get("model_confidence", 0.5)
        historical_accuracy = confidence_params.get("historical_accuracy", 0.5)
        
        # Weighted confidence score (0-1)
        confidence_score = (
            (market_liquidity * self.config["market_liquidity_weight"]) +
            (model_confidence * self.config["model_confidence_weight"]) +
            (historical_accuracy * self.config["historical_accuracy_weight"])
        )
        
        # Adjusted edge - raw edge scaled by confidence
        adjusted_edge = raw_edge * confidence_score
        
        # Calculate Kelly criterion for optimal position sizing
        kelly_fraction = 0
        if raw_edge > 0:
            odds = (1 - market_data["probability"]) / market_data["probability"]
            p = model_data["probability"]
            kelly_fraction = p - (1-p)/odds
            
            # Cap kelly for risk management
            kelly_fraction = max(0, min(0.5, kelly_fraction))
        
        return {
            "raw_edge": raw_edge,
            "confidence_score": confidence_score,
            "adjusted_edge": adjusted_edge,
            "kelly_fraction": kelly_fraction,
            "recommendation": self.generate_recommendation(adjusted_edge, confidence_score)
        }
    
    def generate_recommendation(self, edge: float, confidence: float) -> Dict:
        """
        Generate trading recommendation based on edge and confidence
        
        Args:
            edge: Adjusted edge value
            confidence: Confidence score
            
        Returns:
            Recommendation dictionary
        """
        edge_abs = abs(edge)
        edge_sign = np.sign(edge)
        
        # No significant edge
        if edge_abs < 0.02:
            return {
                "action": "HOLD",
                "strength": "weak",
                "reasoning": "No significant difference between market and model probabilities"
            }
        
        # Low confidence
        if confidence < 0.4:
            return {
                "action": "BUY" if edge_sign > 0 else "SELL",
                "strength": "weak",
                "reasoning": "Some edge detected but confidence is low, proceed with caution"
            }
        
        # Medium edge
        if edge_abs < 0.1:
            return {
                "action": "BUY" if edge_sign > 0 else "SELL",
                "strength": "moderate",
                "reasoning": "Moderate edge with sufficient confidence"
            }
        
        # Strong edge with good confidence
        return {
            "action": "BUY" if edge_sign > 0 else "SELL",
            "strength": "strong",
            "reasoning": "Strong edge with good confidence, significant mispricing detected"
        }
    
    def compare_market_to_model(self, market_data: Dict, model_data: Dict, additional_params: Dict = None) -> Dict:
        """
        Performs a comprehensive comparison between market and model data
        
        Args:
            market_data: Market probability data
            model_data: Model probability data
            additional_params: Additional parameters
            
        Returns:
            Comprehensive comparison
        """
        additional_params = additional_params or {}
        
        # Basic validation
        if not market_data or not model_data:
            raise ValueError("Market and model data are required")
        
        if not isinstance(market_data.get("probability"), (int, float)) or not isinstance(model_data.get("probability"), (int, float)):
            raise ValueError("Probability values must be numbers")
        
        # Calculate metrics
        metrics = self.calculate_comparison_metrics(market_data, model_data)
        
        # Calculate edge
        edge = self.calculate_edge(
            market_data, 
            model_data,
            {
                "market_liquidity": additional_params.get("market_liquidity", 0.5),
                "model_confidence": additional_params.get("model_confidence", 0.5),
                "historical_accuracy": additional_params.get("historical_accuracy", 0.5),
            }
        )
        
        # Generate report
        return {
            "timestamp": datetime.now().isoformat(),
            "market": {
                "probability": market_data["probability"],
                "implied_odds": f"{(1 / market_data['probability']):.2f}",
                "source": market_data.get("source", "unknown"),
                "timestamp": market_data.get("timestamp", None)
            },
            "model": {
                "probability": model_data["probability"],
                "confidence": model_data.get("confidence", "unknown"),
                "methodology": model_data.get("methodology", "unknown"),
                "timestamp": model_data.get("timestamp", None)
            },
            "comparison": {
                **metrics,
                "edge": edge,
                "opportunity": metrics["divergence_significance"] != "negligible",
                "summary": (
                    f"The model {'favors' if edge['recommendation']['action'] == 'BUY' else 'opposes'} "
                    f"this outcome with {edge['recommendation']['strength']} conviction, showing a "
                    f"{metrics['absolute_divergence']*100:.1f}% divergence from market price."
                )
            }
        }


# Example usage when run directly
if __name__ == "__main__":
    # Create comparison model
    comparison_model = MarketModelComparison()
    
    # Sample data
    market_data = {
        "probability": 0.65,
        "source": "manifold",
        "timestamp": datetime.now().isoformat()
    }
    
    model_data = {
        "probability": 0.72,
        "confidence": 0.8,
        "methodology": "LSTM time series forecast",
        "timestamp": datetime.now().isoformat()
    }
    
    # Run comparison
    result = comparison_model.compare_market_to_model(market_data, model_data)
    
    # Print result
    print(f"Market probability: {market_data['probability']}")
    print(f"Model probability: {model_data['probability']}")
    print(f"Divergence: {result['comparison']['absolute_divergence']:.4f}")
    print(f"Recommendation: {result['comparison']['edge']['recommendation']['action']} ({result['comparison']['edge']['recommendation']['strength']})")
    print(f"Summary: {result['comparison']['summary']}")