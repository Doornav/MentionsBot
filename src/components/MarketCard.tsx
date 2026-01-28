import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ProbabilityGauge } from "@/components/ProbabilityGauge";
import { EdgeIndicator } from "@/components/EdgeIndicator";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { Market } from "@/types/market";
import { formatCurrency } from "@/data/mockMarkets";
import { cn } from "@/lib/utils";
import { Clock, TrendingUp } from "lucide-react";

interface MarketCardProps {
  market: Market;
  className?: string;
}

export function MarketCard({ market, className }: MarketCardProps) {
  const daysUntilExpiration = Math.ceil(
    (new Date(market.expirationDate).getTime() - Date.now()) / (1000 * 60 * 60 * 24)
  );

  return (
    <Link to={`/market/${market.id}`}>
      <Card
        className={cn(
          "group relative overflow-hidden transition-all duration-200",
          "hover:border-primary/50 hover:shadow-lg hover:shadow-primary/5",
          "cursor-pointer bg-card",
          className
        )}
      >
        <CardContent className="p-4">
          {/* Header */}
          <div className="flex items-start justify-between gap-2 mb-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <Badge variant="secondary" className="text-xs font-mono">
                  {market.ticker}
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {market.category}
                </Badge>
              </div>
              <h3 className="font-semibold text-sm leading-tight line-clamp-2 group-hover:text-primary transition-colors">
                {market.title}
              </h3>
            </div>
            <ConfidenceBadge confidence={market.confidence} />
          </div>

          {/* Probability Gauges */}
          <div className="flex items-center justify-between gap-4 mb-4">
            <ProbabilityGauge
              value={market.marketProbability}
              label="Market"
              change24h={market.marketProbabilityChange24h}
              size="sm"
            />
            <div className="flex-1 flex items-center justify-center">
              <EdgeIndicator edge={market.edge} direction={market.edgeDirection} />
            </div>
            <ProbabilityGauge
              value={market.modelProbability}
              label="Model"
              change24h={market.modelProbabilityChange24h}
              size="sm"
            />
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between text-xs text-muted-foreground border-t border-border/50 pt-3">
            <div className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              <span className="font-mono">{formatCurrency(market.volume24h)}</span>
              <span>24h vol</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              <span>{daysUntilExpiration}d</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
