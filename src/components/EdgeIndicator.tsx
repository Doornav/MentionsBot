import { cn } from "@/lib/utils";
import { formatEdge } from "@/data/mockMarkets";

interface EdgeIndicatorProps {
  edge: number;
  direction: "BUY_YES" | "BUY_NO" | "HOLD";
  className?: string;
}

export function EdgeIndicator({ edge, direction, className }: EdgeIndicatorProps) {
  const absEdge = Math.abs(edge);
  const isSignificant = absEdge > 0.02;

  return (
    <div className={cn("flex flex-col items-center gap-1", className)}>
      <div
        className={cn(
          "px-3 py-1.5 rounded-md font-mono text-sm font-semibold",
          isSignificant
            ? edge > 0
              ? "bg-bullish/20 text-bullish"
              : "bg-bearish/20 text-bearish"
            : "bg-muted text-muted-foreground"
        )}
      >
        {formatEdge(edge)}
      </div>
      <span
        className={cn(
          "text-xs font-medium uppercase tracking-wider",
          direction === "BUY_YES" && "text-bullish",
          direction === "BUY_NO" && "text-bearish",
          direction === "HOLD" && "text-muted-foreground"
        )}
      >
        {direction.replace("_", " ")}
      </span>
    </div>
  );
}
