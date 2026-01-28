import { useState } from "react";
import { Header } from "@/components/Header";
import { MarketCard } from "@/components/MarketCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { mockMarkets } from "@/data/mockMarkets";
import { Market } from "@/types/market";
import { cn } from "@/lib/utils";
import { ArrowUpDown, Filter, RefreshCw, Zap } from "lucide-react";

type SortField = "edge" | "volume" | "expiration";

const Index = () => {
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [sortField, setSortField] = useState<SortField>("edge");

  const categories = ["all", ...new Set(mockMarkets.map((m) => m.category))];

  const filteredMarkets = mockMarkets
    .filter((m) => activeCategory === "all" || m.category === activeCategory)
    .sort((a, b) => {
      switch (sortField) {
        case "edge":
          return Math.abs(b.edge) - Math.abs(a.edge);
        case "volume":
          return b.volume24h - a.volume24h;
        case "expiration":
          return new Date(a.expirationDate).getTime() - new Date(b.expirationDate).getTime();
        default:
          return 0;
      }
    });

  const topEdgeMarkets = [...mockMarkets]
    .sort((a, b) => Math.abs(b.edge) - Math.abs(a.edge))
    .slice(0, 3);

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container py-6 space-y-6">
        {/* Hero Section */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold">US Politics Forecast Engine</h1>
            <Badge variant="outline" className="font-mono text-xs">
              KALSHI
            </Badge>
          </div>
          <p className="text-muted-foreground">
            Compare market-implied probability with model-implied probability. Find edge in prediction markets.
          </p>
        </div>

        {/* Top Edge Opportunities */}
        <div className="p-4 rounded-lg border border-primary/30 bg-primary/5">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4 text-primary" />
            <span className="font-semibold text-sm">Top Edge Opportunities</span>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            {topEdgeMarkets.map((market) => (
              <div
                key={market.id}
                className="flex items-center justify-between p-2 rounded bg-background/50"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <Badge variant="secondary" className="font-mono text-xs shrink-0">
                    {market.ticker}
                  </Badge>
                  <span className="text-sm truncate">{market.title.slice(0, 30)}...</span>
                </div>
                <span
                  className={cn(
                    "font-mono text-sm font-semibold shrink-0",
                    market.edge > 0 ? "text-bullish" : "text-bearish"
                  )}
                >
                  {market.edge > 0 ? "+" : ""}
                  {(market.edge * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Filters & Controls */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          {/* Category Tabs */}
          <Tabs value={activeCategory} onValueChange={setActiveCategory}>
            <TabsList className="h-9">
              {categories.map((cat) => (
                <TabsTrigger key={cat} value={cat} className="text-xs capitalize">
                  {cat}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>

          {/* Sort & Actions */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <ArrowUpDown className="w-3 h-3" />
              Sort:
            </div>
            <Button
              variant={sortField === "edge" ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setSortField("edge")}
              className="text-xs h-8"
            >
              Edge
            </Button>
            <Button
              variant={sortField === "volume" ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setSortField("volume")}
              className="text-xs h-8"
            >
              Volume
            </Button>
            <Button
              variant={sortField === "expiration" ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setSortField("expiration")}
              className="text-xs h-8"
            >
              Expiry
            </Button>
            <div className="w-px h-6 bg-border mx-2" />
            <Button variant="outline" size="sm" className="text-xs h-8 gap-1">
              <RefreshCw className="w-3 h-3" />
              Refresh
            </Button>
          </div>
        </div>

        {/* Market Grid */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredMarkets.map((market) => (
            <MarketCard key={market.id} market={market} />
          ))}
        </div>

        {/* Footer Stats */}
        <div className="flex items-center justify-between text-xs text-muted-foreground pt-4 border-t border-border/50">
          <div className="flex items-center gap-4">
            <span>
              <span className="font-mono">{mockMarkets.length}</span> markets tracked
            </span>
            <span>
              Last updated: <span className="font-mono">just now</span>
            </span>
          </div>
          <span className="font-mono">v0.1.0-beta</span>
        </div>
      </main>
    </div>
  );
};

export default Index;
