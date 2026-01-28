import { useParams, Link } from "react-router-dom";
import { Header } from "@/components/Header";
import { ProbabilityGauge } from "@/components/ProbabilityGauge";
import { EdgeIndicator } from "@/components/EdgeIndicator";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { EvidenceCard } from "@/components/EvidenceCard";
import { ForecastSummary } from "@/components/ForecastSummary";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  getMarketById,
  getEvidenceForMarket,
  getForecastForMarket,
  formatCurrency,
  formatPercent,
} from "@/data/mockMarkets";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  Calendar,
  Clock,
  ExternalLink,
  TrendingUp,
  Volume2,
  Wallet,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";

const MarketDetail = () => {
  const { id } = useParams<{ id: string }>();
  const market = id ? getMarketById(id) : undefined;
  const evidence = id ? getEvidenceForMarket(id) : [];
  const forecast = id ? getForecastForMarket(id) : undefined;

  if (!market) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <main className="container py-12 text-center">
          <h1 className="text-2xl font-bold mb-4">Market Not Found</h1>
          <Link to="/">
            <Button variant="outline">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Dashboard
            </Button>
          </Link>
        </main>
      </div>
    );
  }

  const daysUntilExpiration = Math.ceil(
    (new Date(market.expirationDate).getTime() - Date.now()) / (1000 * 60 * 60 * 24)
  );

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container py-6 space-y-6">
        {/* Back Button */}
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </Link>

        {/* Market Header */}
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="font-mono text-sm">
              {market.ticker}
            </Badge>
            <Badge variant="outline">{market.category}</Badge>
            <Badge
              variant="outline"
              className={cn(
                market.status === "active" && "border-bullish text-bullish",
                market.status === "closed" && "border-muted text-muted-foreground",
                market.status === "resolved" && "border-primary text-primary"
              )}
            >
              {market.status.toUpperCase()}
            </Badge>
            <ConfidenceBadge confidence={market.confidence} />
          </div>

          <h1 className="text-2xl font-bold leading-tight">{market.title}</h1>
          {market.subtitle && (
            <p className="text-muted-foreground">{market.subtitle}</p>
          )}

          {/* Quick Stats */}
          <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <Calendar className="w-4 h-4" />
              Expires in {daysUntilExpiration} days
            </span>
            <span className="flex items-center gap-1">
              <TrendingUp className="w-4 h-4" />
              {formatCurrency(market.volume24h)} 24h volume
            </span>
            <span className="flex items-center gap-1">
              <Wallet className="w-4 h-4" />
              {formatCurrency(market.openInterest)} open interest
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              Updated {formatDistanceToNow(new Date(market.lastUpdated), { addSuffix: true })}
            </span>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left Column - Probability Comparison */}
          <div className="lg:col-span-2 space-y-6">
            {/* Probability Card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                  Probability Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-around gap-4 py-4">
                  <ProbabilityGauge
                    value={market.marketProbability}
                    label="Market"
                    change24h={market.marketProbabilityChange24h}
                    size="lg"
                  />
                  <EdgeIndicator edge={market.edge} direction={market.edgeDirection} />
                  <ProbabilityGauge
                    value={market.modelProbability}
                    label="Model"
                    change24h={market.modelProbabilityChange24h}
                    size="lg"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Evidence Feed */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                  Evidence Feed ({evidence.length})
                </CardTitle>
                <Button variant="outline" size="sm" className="text-xs">
                  <ExternalLink className="w-3 h-3 mr-1" />
                  View All Sources
                </Button>
              </CardHeader>
              <CardContent>
                {evidence.length > 0 ? (
                  <div className="space-y-3">
                    {evidence.map((ev) => (
                      <EvidenceCard key={ev.id} evidence={ev} />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No evidence items for this market yet.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Forecast */}
          <div className="space-y-6">
            {forecast ? (
              <ForecastSummary forecast={forecast} />
            ) : (
              <Card>
                <CardContent className="py-8 text-center text-muted-foreground">
                  No forecast available for this market.
                </CardContent>
              </Card>
            )}

            {/* External Links */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                  External Links
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full justify-start" asChild>
                  <a
                    href={`https://kalshi.com/markets/${market.ticker.toLowerCase()}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <ExternalLink className="w-4 h-4 mr-2" />
                    View on Kalshi
                  </a>
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default MarketDetail;
