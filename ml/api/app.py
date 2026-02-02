"""
FastAPI application for ML model serving
"""

import os
from typing import Dict, List, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Import models
from ml.models.sentiment.sentiment_model import SentimentModel
from ml.models.forecast.time_series_model import TimeSeriesModel
from ml.models.comparison.market_model_comparison import MarketModelComparison

app = FastAPI(
    title="Market Insights Engine API",
    description="API for forecast and evidence engine",
    version="0.1.0",
)

# Initialize models
sentiment_model = SentimentModel()
time_series_model = TimeSeriesModel()
comparison_model = MarketModelComparison()

# Model input/output schemas
class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text content to analyze")
    topic: str = Field(..., description="Topic for relevance calculation")
    topic_keywords: Optional[Dict[str, float]] = Field(
        None, description="Additional topic keywords and their weights"
    )


class TextAnalysisResponse(BaseModel):
    sentiment: Dict = Field(..., description="Sentiment analysis results")
    relevance: Dict = Field(..., description="Relevance analysis results")
    confidence: float = Field(..., description="Overall confidence score")
    key_phrases: List[str] = Field(..., description="Key phrases extracted")
    timestamp: str = Field(..., description="Analysis timestamp")


class TimeSeriesRequest(BaseModel):
    data: List[float] = Field(..., description="Historical time series data")
    forecast_steps: Optional[int] = Field(None, description="Number of steps to forecast")
    config: Optional[Dict] = Field(None, description="Model configuration overrides")


class TimeSeriesResponse(BaseModel):
    forecast: List[float] = Field(..., description="Forecasted values")
    confidence: float = Field(..., description="Forecast confidence")
    timestamp: str = Field(..., description="Forecast timestamp")


class MarketDataRequest(BaseModel):
    market_probability: float = Field(..., description="Market implied probability")
    model_probability: float = Field(..., description="Model forecast probability")
    market_source: Optional[str] = Field(None, description="Source of market data")
    model_confidence: Optional[float] = Field(None, description="Model confidence")
    market_liquidity: Optional[float] = Field(None, description="Market liquidity")
    historical_accuracy: Optional[float] = Field(None, description="Historical accuracy of model")


class MarketDataResponse(BaseModel):
    comparison: Dict = Field(..., description="Comparison metrics")
    recommendation: Dict = Field(..., description="Trading recommendation")
    timestamp: str = Field(..., description="Analysis timestamp")


# Routes
@app.get("/")
def read_root():
    return {
        "name": "Market Insights Engine API",
        "version": "0.1.0",
        "endpoints": [
            "/sentiment/analyze",
            "/forecast/predict",
            "/comparison/analyze",
            "/health"
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "sentiment": "loaded" if sentiment_model is not None else "not_loaded",
            "time_series": "loaded" if time_series_model is not None else "not_loaded",
            "comparison": "loaded" if comparison_model is not None else "not_loaded"
        }
    }


@app.post("/sentiment/analyze", response_model=TextAnalysisResponse)
def analyze_text(request: TextAnalysisRequest):
    try:
        result = sentiment_model.analyze_text(
            request.text,
            request.topic,
            {"topic_keywords": request.topic_keywords}
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@app.post("/forecast/predict", response_model=TimeSeriesResponse)
def predict_forecast(request: TimeSeriesRequest, background_tasks: BackgroundTasks):
    try:
        # Check if the model is trained
        if time_series_model.model is None:
            # Train in the background if no model exists
            background_tasks.add_task(time_series_model.train, request.data)
            raise HTTPException(status_code=409, detail="Model is being trained, please try again soon.")
        
        # Make forecast
        forecast_steps = request.forecast_steps or time_series_model.config["forecast_horizon"]
        forecast_data = time_series_model.multi_step_forecast(request.data, steps=forecast_steps)
        
        # Calculate a simple confidence based on validation loss
        confidence = 0.8  # Default placeholder
        
        return {
            "forecast": forecast_data,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")


@app.post("/forecast/train", response_model=Dict)
def train_forecast(request: TimeSeriesRequest):
    try:
        # Train the model
        history = time_series_model.train(request.data, request.config)
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "history": {
                "train_loss": history["train_loss"][-1],
                "val_loss": history["val_loss"][-1],
                "epochs": len(history["train_loss"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.post("/comparison/analyze", response_model=MarketDataResponse)
def compare_probabilities(request: MarketDataRequest):
    try:
        # Prepare input data structures
        market_data = {
            "probability": request.market_probability,
            "source": request.market_source,
            "timestamp": datetime.now().isoformat()
        }
        
        model_data = {
            "probability": request.model_probability,
            "confidence": request.model_confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        # Optional parameters for edge calculation
        additional_params = {}
        if request.market_liquidity is not None:
            additional_params["market_liquidity"] = request.market_liquidity
        
        if request.model_confidence is not None:
            additional_params["model_confidence"] = request.model_confidence
        
        if request.historical_accuracy is not None:
            additional_params["historical_accuracy"] = request.historical_accuracy
        
        # Perform comparison
        result = comparison_model.compare_market_to_model(market_data, model_data, additional_params)
        
        return {
            "comparison": result["comparison"],
            "recommendation": result["comparison"]["edge"]["recommendation"],
            "timestamp": result["timestamp"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing probabilities: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)