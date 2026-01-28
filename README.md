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
└── backend/            # ML and data processing backend
    ├── api/            # API endpoints
    │   ├── controllers/  # Request handlers
    │   └── routes/       # API route definitions
    ├── models/         # ML model definitions
    │   ├── probability/  # Probability models
    │   ├── forecast/     # Forecasting models
    │   ├── sentiment/    # Sentiment analysis models
    │   └── comparison/   # Market vs. model comparison
    ├── services/       # Business logic services
    ├── utils/          # Utility functions
    └── data/           # Data processing
        ├── sources/      # Data source adapters
        ├── processors/   # Data processing pipelines
        └── storage/      # Data storage solutions
```

## Technologies Used

### Frontend
- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

### Backend
- Node.js with Express
- TensorFlow.js for ML models
- Natural language processing libraries
- Mathematical utilities for probability analysis

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
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

3. Install backend dependencies
```bash
cd backend
npm install
cd ..
```

### Running the Application

#### Development Mode

1. Start the frontend development server
```bash
npm run dev
```

2. Start the backend server
```bash
cd backend
npm run dev
```

#### Production Build

1. Build the frontend
```bash
npm run build
```

2. Build the backend
```bash
cd backend
npm run build
```

## ML Architecture

Our ML architecture consists of several specialized models:

1. **Time Series Forecasting Models** - Predict future probabilities based on historical data
2. **Sentiment Analysis Models** - Analyze news and social media for market-relevant sentiment
3. **Market-Model Comparison** - Identify discrepancies between market prices and model forecasts

Each model is designed to work independently and in conjunction with others to provide comprehensive insights.

## Future Development

- Integration with live prediction market APIs
- Enhanced ML models with reinforcement learning
- User customization of model parameters
- Mobile application

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.