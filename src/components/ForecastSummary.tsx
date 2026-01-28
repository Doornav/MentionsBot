import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Forecast } from "@/types/market";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { cn } from "@/lib/utils";
import { AlertCircle, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { formatPercent, formatEdge } from "@/data/mockMarkets";

interface ForecastSummaryProps {
  forecast: Forecast;
  className?: string;
}

export function ForecastSummary({ forecast, className }: ForecastSummaryProps) {
  const getRecommendationIcon = () => {
    switch (forecast.recommendation) {
      case "BUY_YES":
        return <TrendingUp className="w-5 h-5 text-bullish" />;
      case "BUY_NO":
        return <TrendingDown className="w-5 h-5 text-bearish" />;
      case "HOLD":
        return <Minus className="w-5 h-5 text-muted-foreground" />;
    }
  };

  return (
    <Card className={cn("bg-card", className)}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
            Model Forecast
          </CardTitle>
          <ConfidenceBadge confidence={forecast.confidence} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Recommendation */}
        <div className="flex items-center gap-3 p-3 rounded-lg bg-accent/50">
          {getRecommendationIcon()}
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className={cn(
                "font-semibold",
                forecast.recommendation === "BUY_YES" && "text-bullish",
                forecast.recommendation === "BUY_NO" && "text-bearish",
                forecast.recommendation === "HOLD" && "text-muted-foreground"
              )}>
                {forecast.recommendation.replace("_", " ")}
              </span>
              {forecast.edgeSignificant && (
                <Badge variant="outline" className="text-xs border-primary text-primary">
                  SIGNIFICANT
                </Badge>
              )}
            </div>
            <div className="text-sm text-muted-foreground">
              Edge: <span className="font-mono">{formatEdge(forecast.edge)}</span>
            </div>
          </div>
        </div>

        {/* What Changed */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium">
            <AlertCircle className="w-4 h-4 text-primary" />
            What changed in 24h
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {forecast.whatChanged24h}
          </p>
        </div>

        {/* Key Drivers */}
        <div className="space-y-2">
          <div className="text-sm font-medium">Key Drivers</div>
          <ul className="space-y-1">
            {forecast.keyDrivers.map((driver, idx) => (
              <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                <span className="text-primary">•</span>
                {driver}
              </li>
            ))}
          </ul>
        </div>

        {/* Evidence Summary */}
        <div className="flex items-center gap-4 pt-2 border-t border-border/50 text-xs text-muted-foreground">
          <span className="font-mono">{forecast.evidenceCount} items</span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-bullish" />
            {forecast.bullishCount} YES
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-bearish" />
            {forecast.bearishCount} NO
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-neutral" />
            {forecast.neutralCount} NEUTRAL
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
