// Core market and evidence types for the Forecast Engine

export type Direction = "YES" | "NO" | "NEUTRAL";
export type Strength = "WEAK" | "MEDIUM" | "STRONG";
export type Confidence = "LOW" | "MEDIUM" | "HIGH";

export interface Market {
  id: string;
  ticker: string;
  title: string;
  subtitle?: string;
  category: string;
  expirationDate: string;
  status: "active" | "closed" | "resolved";
  
  // Market-implied probability (from Kalshi)
  marketProbability: number;
  marketProbabilityChange24h: number;
  
  // Model-implied probability (our forecast)
  modelProbability: number;
  modelProbabilityChange24h: number;
  
  // Edge = model - market
  edge: number;
  edgeDirection: "BUY_YES" | "BUY_NO" | "HOLD";
  
  // Confidence in our model
  confidence: Confidence;
  
  // Volume and liquidity
  volume24h: number;
  openInterest: number;
  
  // Last updated
  lastUpdated: string;
}

export interface Evidence {
  id: string;
  marketId: string;
  
  // Source info
  source: string;
  sourceType: "news" | "twitter" | "official" | "poll";
  sourceCredibility: number; // 0-1
  url: string;
  
  // Content
  title: string;
  summary: string;
  publishedAt: string;
  
  // Scoring
  direction: Direction;
  strength: Strength;
  relevanceScore: number; // 0-1
  
  // Metadata
  author?: string;
  isVerified: boolean;
}

export interface Forecast {
  id: string;
  marketId: string;
  
  // Probabilities
  marketPrior: number;
  modelPosterior: number;
  confidence: Confidence;
  
  // Edge analysis
  edge: number;
  edgeSignificant: boolean;
  recommendation: "BUY_YES" | "BUY_NO" | "HOLD";
  
  // Evidence summary
  evidenceCount: number;
  bullishCount: number;
  bearishCount: number;
  neutralCount: number;
  
  // Explainability
  whatChanged24h: string;
  keyDrivers: string[];
  
  // Timestamps
  generatedAt: string;
  validUntil: string;
}

export interface Snapshot {
  id: string;
  marketId: string;
  timestamp: string;
  
  marketProbability: number;
  modelProbability: number;
  edge: number;
  confidence: Confidence;
  
  evidenceIds: string[];
}

// Filter and sort options
export type MarketSortField = "edge" | "confidence" | "volume" | "expiration" | "probability";
export type SortDirection = "asc" | "desc";

export interface MarketFilters {
  category?: string;
  status?: Market["status"];
  minEdge?: number;
  maxExpiration?: string;
  confidence?: Confidence[];
}
