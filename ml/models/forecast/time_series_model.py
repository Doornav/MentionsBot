"""
Time Series Forecasting Model

This model uses historical data to forecast future probabilities
using PyTorch-based LSTM models.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    """LSTM-based time series forecasting model"""
    
    def __init__(self, 
                input_dim: int = 1, 
                hidden_dim: int = 64, 
                num_layers: int = 2, 
                output_dim: int = 1, 
                dropout: float = 0.2):
        """
        Initialize LSTM model
        
        Args:
            input_dim: Input dimension (features)
            hidden_dim: Hidden dimension of LSTM layers
            num_layers: Number of LSTM layers
            output_dim: Output dimension (typically 1 for forecasting)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Get the last time step's output
        y_pred = self.fc(lstm_out[:, -1, :])
        
        return y_pred


class TimeSeriesModel:
    def __init__(self, config: Dict = None):
        """
        Initialize the time series forecasting model
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            "sequence_length": 14,      # Number of days to use for sequence input
            "forecast_horizon": 7,      # Number of days to forecast ahead
            "hidden_layers": [64, 32],  # Architecture of LSTM
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
            **(config or {})
        }
        
        self.model = None
        self.metadata = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def preprocess_data(self, data: List[float]) -> Dict:
        """
        Preprocess time series data
        
        Args:
            data: Array of time series values
            
        Returns:
            Processed data for model input
        """
        # Ensure data is a numpy array
        data = np.array(data, dtype=np.float32)
        
        # Normalize the data
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val) if max_val > min_val else data
        
        # Create sequences for training
        sequences = []
        targets = []
        
        for i in range(len(normalized_data) - self.config["sequence_length"] - self.config["forecast_horizon"] + 1):
            # Extract sequence
            seq = normalized_data[i:i + self.config["sequence_length"]]
            # Extract target (value after forecast_horizon)
            target = normalized_data[i + self.config["sequence_length"] + self.config["forecast_horizon"] - 1]
            
            sequences.append(seq)
            targets.append(target)
        
        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        return {
            "sequences": sequences,
            "targets": targets,
            "min": min_val,
            "max": max_val
        }
    
    def build_model(self) -> LSTMModel:
        """
        Build the LSTM model architecture
        
        Returns:
            LSTM model
        """
        # Create model with config
        model = LSTMModel(
            input_dim=1,
            hidden_dim=self.config["hidden_layers"][0],
            num_layers=len(self.config["hidden_layers"]),
            output_dim=1,
            dropout=0.2
        )
        
        model.to(self.device)
        return model
    
    def train(self, data: List[float], training_config: Dict = None) -> Dict:
        """
        Train the model with historical data
        
        Args:
            data: Historical time series data
            training_config: Training configuration
            
        Returns:
            Training history
        """
        # Set training configuration
        config = {
            "epochs": self.config["epochs"],
            "batch_size": self.config["batch_size"],
            "validation_split": self.config["validation_split"],
            **(training_config or {})
        }
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Extract data
        sequences = processed_data["sequences"]
        targets = processed_data["targets"]
        
        # Reshape sequences for LSTM (batch_size, seq_len, features)
        sequences = sequences.reshape(-1, self.config["sequence_length"], 1)
        
        # Calculate split indices
        train_size = int(len(sequences) * (1 - config["validation_split"]))
        
        # Split data
        X_train = sequences[:train_size]
        y_train = targets[:train_size]
        X_val = sequences[train_size:]
        y_val = targets[train_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["batch_size"],
            shuffle=False
        )
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # Training loop
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                # Move to device
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)
            
            print(f"Epoch {epoch+1}/{config['epochs']}: loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save metadata for denormalization
        self.metadata = {
            "min": processed_data["min"],
            "max": processed_data["max"]
        }
        
        return history
    
    def forecast(self, recent_data: List[float]) -> List[float]:
        """
        Make a forecast based on recent data
        
        Args:
            recent_data: Recent time series data
            
        Returns:
            Forecasted values
        """
        # Ensure model is built
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure we have enough data
        if len(recent_data) < self.config["sequence_length"]:
            raise ValueError(f"Need at least {self.config['sequence_length']} data points, got {len(recent_data)}")
        
        # Normalize data using saved metadata
        normalized_data = (np.array(recent_data) - self.metadata["min"]) / (self.metadata["max"] - self.metadata["min"])
        
        # Use the most recent window of data
        input_sequence = normalized_data[-self.config["sequence_length"]:]
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).reshape(1, self.config["sequence_length"], 1).to(self.device)
        
        # Switch to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            forecast_normalized = self.model(input_tensor).cpu().numpy().flatten()
        
        # Denormalize the output
        forecast = forecast_normalized * (self.metadata["max"] - self.metadata["min"]) + self.metadata["min"]
        
        return forecast.tolist()
    
    def multi_step_forecast(self, recent_data: List[float], steps: int = None) -> List[float]:
        """
        Make a multi-step forecast
        
        Args:
            recent_data: Recent time series data
            steps: Number of steps to forecast (defaults to forecast_horizon)
            
        Returns:
            List of forecasted values
        """
        steps = steps or self.config["forecast_horizon"]
        
        # Start with the recent data
        forecasted_data = list(recent_data)
        
        # Perform iterative forecasting
        for _ in range(steps):
            # Get the latest sequence
            latest_sequence = forecasted_data[-self.config["sequence_length"]:]
            
            # Make a one-step forecast
            next_value = self.forecast(latest_sequence)[0]
            
            # Add to the forecasted data
            forecasted_data.append(next_value)
        
        # Return only the newly forecasted values
        return forecasted_data[-steps:]
    
    def save_model(self, path: str) -> bool:
        """
        Save the model to a file
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metadata': self.metadata
        }, path)
        
        return True
    
    def load_model(self, path: str) -> LSTMModel:
        """
        Load a model from a file
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update config
        self.config = checkpoint['config']
        
        # Create model
        self.model = self.build_model()
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load metadata
        self.metadata = checkpoint['metadata']
        
        # Set to eval mode
        self.model.eval()
        
        return self.model


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate sample data: sine wave + noise
    x = np.linspace(0, 4 * np.pi, 200)
    data = 0.5 + 0.4 * np.sin(x) + 0.1 * np.random.randn(200)
    
    # Create and train model
    model = TimeSeriesModel({
        "sequence_length": 20,
        "forecast_horizon": 10,
        "epochs": 100
    })
    
    history = model.train(data)
    
    # Generate forecast
    forecast = model.multi_step_forecast(data[:150], steps=30)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Actual')
    plt.plot(range(150, 150 + len(forecast)), forecast, label='Forecast')
    plt.legend()
    plt.title('Time Series Forecast')
    plt.savefig('forecast.png')
    plt.close()