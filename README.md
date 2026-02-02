# Market Insights Engine

A forecast + evidence engine that helps users compare market-implied probability with model-implied probability for prediction markets.

## Project Overview

This platform provides powerful insights by comparing market-implied probabilities from prediction markets with our proprietary ML models. It focuses on 10-20 active prediction markets within specific verticals (e.g., US politics or macroeconomics).

### Key Features

- Compare market-implied vs model-implied probabilities
- View evidence supporting forecasts
- Track probability trends over time
- Analyze market mispricing opportunities

## Project Structure

```
/
├── src/                # Frontend React application
│   ├── components/     # UI components
│   ├── pages/          # Page components
│   ├── data/           # Mock data
│   ├── hooks/          # Custom React hooks
│   ├── lib/            # Utility functions
│   └── types/          # TypeScript type definitions
│
├── ml/                 # Python-based ML models
│   ├── api/            # FastAPI service
│   ├── models/         # ML model implementations
│   │   ├── forecast/     # Time series forecasting models
│   │   ├── sentiment/    # Sentiment analysis models
│   │   └── comparison/   # Market vs. model comparison logic
│   ├── utils/          # ML utility functions
│   └── data/           # Data processing pipeline
│
└── backend/            # Node.js API service (deprecated)
```

## Technologies Used

### Frontend
- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

### ML Backend
- Python 3.9+
- PyTorch for ML models
- Transformers for NLP
- FastAPI for API endpoints
- Pandas for data manipulation
- NumPy for numerical operations

## Getting Started

### Prerequisites

- Node.js (v16 or higher) for frontend
- Python 3.9+ for ML service
- npm or yarn

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd market-insights-engine
```

2. Install frontend dependencies
```bash
npm install
```

3. Install ML dependencies
```bash
cd ml
pip install -r requirements.txt
cd ..
```

### Running the Application

#### Development Mode

1. Start the frontend development server
```bash
npm run dev
```

2. Start the ML service
```bash
cd ml
python run_ml_service.py --debug
```

#### Production Build

1. Build the frontend
```bash
npm run build
```

2. Run the ML service in production mode
```bash
cd ml
python run_ml_service.py
```

## ML Architecture

Our ML architecture consists of several specialized PyTorch-based models:

1. **Time Series Forecasting Models** - LSTM neural networks to predict future probabilities based on historical data
2. **Sentiment Analysis Models** - Transformer-based models to analyze news and social media for market-relevant sentiment
3. **Market-Model Comparison** - Probabilistic analysis to identify discrepancies between market prices and model forecasts

### Why PyTorch?

We chose PyTorch for our ML implementation for several reasons:

1. **Advanced ML Capabilities**: PyTorch provides state-of-the-art deep learning capabilities for time series forecasting and NLP
2. **Research-Friendly**: Easier experimentation with model architectures and hyperparameters
3. **Production-Ready**: Efficient model deployment with TorchScript and ONNX integration
4. **Extensive Ecosystem**: Access to pre-trained models like BERT and GPT for advanced NLP
5. **Strong Community Support**: Well-maintained libraries and active development

## API Endpoints

Our ML service provides several REST API endpoints:

- **POST /sentiment/analyze** - Analyze text for sentiment and relevance
- **POST /forecast/predict** - Generate time series forecasts
- **POST /comparison/analyze** - Compare market and model probabilities
- **GET /health** - Check ML service health

## Future Development

- Integration with live prediction market APIs
- Enhanced ML models with reinforcement learning
- User customization of model parameters
- Mobile application
- Model explainability features