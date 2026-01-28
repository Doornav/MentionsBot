import { cn } from "@/lib/utils";
import { Confidence } from "@/types/market";

interface ConfidenceBadgeProps {
  confidence: Confidence;
  className?: string;
}

export function ConfidenceBadge({ confidence, className }: ConfidenceBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium uppercase tracking-wider",
        confidence === "HIGH" && "bg-strength-strong/20 text-strength-strong",
        confidence === "MEDIUM" && "bg-strength-medium/20 text-strength-medium",
        confidence === "LOW" && "bg-strength-weak/20 text-strength-weak",
        className
      )}
    >
      {confidence}
    </span>
  );
}
