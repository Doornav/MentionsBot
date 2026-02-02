import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class TrumpTweetForecaster:
    def __init__(self):
        """
        Initializes the forecaster. 
        In a real version, this would load your PyTorch/TensorFlow model weights.
        """
        self.reference_data = None
        self.decay_factor = 0.95  # Weights recent tweets higher than old ones
    
    def ingest_data(self, tweets: List[Dict]):
        """
        Loads historical tweets into memory.
        
        Args:
            tweets: List of dicts e.g., [{'text': 'MAGA!', 'created_at': '2024-10-01'}]
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(tweets)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Sort by date (newest first)
        self.reference_data = df.sort_values('created_at', ascending=False)
        print(f"DEBUG: Ingested {len(df)} tweets.")

    def predict_keyword_probability(self, target_word: str, time_window_hours: int = 24) -> float:
        """
        Calculates the probability of a specific word appearing in the next N hours.
        
        Current Logic: Hybrid (Base Rate + Recency Boost)
        """
        if self.reference_data is None:
            raise ValueError("No data ingested. Run ingest_data() first.")

        target = target_word.lower()
        df = self.reference_data.copy()
        
        # 1. FEATURE ENGINEERING
        # Check if previous tweets contained the target
        df['contains_target'] = df['text'].str.lower().str.contains(target, regex=False)
        
        # 2. CALCULATE BASE RATE (The "Long Term Memory")
        # How often does he say this word in general?
        total_tweets = len(df)
        total_occurrences = df['contains_target'].sum()
        base_rate = total_occurrences / total_tweets if total_tweets > 0 else 0
        
        # 3. CALCULATE RECENCY SIGNAL (The "Short Term Memory")
        # If he said it recently, he's "looping" on that topic.
        # We look at the last 50 tweets.
        recent_window = df.head(50)
        recent_count = recent_window['contains_target'].sum()
        
        # 4. THE "MODEL" (Simplified for now)
        # We combine Base Rate with a Recency Multiplier
        
        # If he said it recently, probability spikes
        if recent_count > 0:
            # Simple logistic growth based on recency count
            recency_boost = 1 + (recent_count * 0.5) 
            probability = min(0.99, base_rate * recency_boost * 10) # *10 is an arbitrary scaling factor for demo
        else:
            # Revert to mean (base rate), slightly decayed
            probability = base_rate
            
        # 5. SATURATION CAP
        # Even if he loves the word, probability rarely exceeds 90% for a single 24h window
        return round(min(probability, 0.90), 4)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Instantiate
    forecaster = TrumpTweetForecaster()
    
    # 2. Mock Data (What your API would provide)
    mock_tweets = [
        {'text': "The economy is doing great, very huge!", 'created_at': '2024-10-25 10:00'},
        {'text': "Crypto is the future, unlike the failing banks.", 'created_at': '2024-10-25 09:30'},
        {'text': "We love Crypto! Bitcoin to the moon!", 'created_at': '2024-10-24 18:00'},
        {'text': "Borders are important.", 'created_at': '2024-10-23 12:00'},
        {'text': "Fake news media!", 'created_at': '2024-10-22 08:00'},
    ]
    
    # 3. Load Data
    forecaster.ingest_data(mock_tweets)
    
    # 4. Run Prediction
    target = "crypto"
    prob = forecaster.predict_keyword_probability(target)
    
    print(f"\nPrediction for keyword '{target}':")
    print(f"Calculated Probability: {prob * 100}%")
    
    # Compare with a word he hasn't said
    target_unused = "aliens"
    prob_unused = forecaster.predict_keyword_probability(target_unused)
    print(f"\nPrediction for keyword '{target_unused}':")
    print(f"Calculated Probability: {prob_unused * 100}%")