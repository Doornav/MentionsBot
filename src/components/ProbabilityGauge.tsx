import { cn } from "@/lib/utils";
import { formatPercent } from "@/data/mockMarkets";

interface ProbabilityGaugeProps {
  value: number;
  label: string;
  change24h?: number;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function ProbabilityGauge({
  value,
  label,
  change24h,
  size = "md",
  className,
}: ProbabilityGaugeProps) {
  const sizeClasses = {
    sm: "w-16 h-16",
    md: "w-24 h-24",
    lg: "w-32 h-32",
  };

  const textSizes = {
    sm: "text-lg",
    md: "text-2xl",
    lg: "text-3xl",
  };

  const labelSizes = {
    sm: "text-[10px]",
    md: "text-xs",
    lg: "text-sm",
  };

  // Calculate color based on probability
  const getColor = (p: number) => {
    if (p >= 0.65) return "hsl(var(--bullish))";
    if (p <= 0.35) return "hsl(var(--bearish))";
    return "hsl(var(--neutral))";
  };

  const circumference = 2 * Math.PI * 42;
  const progress = circumference - value * circumference;

  return (
    <div className={cn("flex flex-col items-center gap-1", className)}>
      <div className={cn("relative", sizeClasses[size])}>
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke="hsl(var(--muted))"
            strokeWidth="8"
          />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke={getColor(value)}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={progress}
            className="transition-all duration-500 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={cn("font-mono font-bold", textSizes[size])}>
            {formatPercent(value, 0)}
          </span>
        </div>
      </div>
      <span className={cn("text-muted-foreground uppercase tracking-wider", labelSizes[size])}>
        {label}
      </span>
      {change24h !== undefined && (
        <span
          className={cn(
            "text-xs font-mono",
            change24h > 0 ? "text-bullish" : change24h < 0 ? "text-bearish" : "text-muted-foreground"
          )}
        >
          {change24h > 0 ? "+" : ""}
          {formatPercent(change24h, 1)}
        </span>
      )}
    </div>
  );
}
