import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Evidence } from "@/types/market";
import { cn } from "@/lib/utils";
import { ExternalLink, CheckCircle2, Twitter, Newspaper, Vote, Building2 } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

interface EvidenceCardProps {
  evidence: Evidence;
  compact?: boolean;
  className?: string;
}

export function EvidenceCard({ evidence, compact = false, className }: EvidenceCardProps) {
  const getSourceIcon = () => {
    switch (evidence.sourceType) {
      case "twitter":
        return <Twitter className="w-3.5 h-3.5" />;
      case "news":
        return <Newspaper className="w-3.5 h-3.5" />;
      case "poll":
        return <Vote className="w-3.5 h-3.5" />;
      case "official":
        return <Building2 className="w-3.5 h-3.5" />;
    }
  };

  const getDirectionStyles = () => {
    switch (evidence.direction) {
      case "YES":
        return "border-l-bullish bg-bullish/5";
      case "NO":
        return "border-l-bearish bg-bearish/5";
      case "NEUTRAL":
        return "border-l-neutral bg-neutral/5";
    }
  };

  const getStrengthStyles = () => {
    switch (evidence.strength) {
      case "STRONG":
        return "bg-strength-strong/20 text-strength-strong";
      case "MEDIUM":
        return "bg-strength-medium/20 text-strength-medium";
      case "WEAK":
        return "bg-strength-weak/20 text-strength-weak";
    }
  };

  return (
    <Card
      className={cn(
        "border-l-4 transition-all hover:shadow-md",
        getDirectionStyles(),
        className
      )}
    >
      <CardContent className={cn("p-3", compact && "p-2")}>
        {/* Header */}
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            {getSourceIcon()}
            <span className="font-medium">{evidence.source}</span>
            {evidence.isVerified && (
              <CheckCircle2 className="w-3 h-3 text-primary" />
            )}
            <span>•</span>
            <span>{formatDistanceToNow(new Date(evidence.publishedAt), { addSuffix: true })}</span>
          </div>
          <a
            href={evidence.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-primary transition-colors"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        {/* Title */}
        <h4 className={cn("font-medium leading-tight mb-2", compact ? "text-sm" : "text-sm")}>
          {evidence.title}
        </h4>

        {/* Summary - hidden in compact mode */}
        {!compact && (
          <p className="text-xs text-muted-foreground leading-relaxed mb-3">
            {evidence.summary}
          </p>
        )}

        {/* Badges */}
        <div className="flex items-center gap-2 flex-wrap">
          <Badge
            variant="outline"
            className={cn(
              "text-xs font-mono",
              evidence.direction === "YES" && "border-bullish text-bullish",
              evidence.direction === "NO" && "border-bearish text-bearish",
              evidence.direction === "NEUTRAL" && "border-neutral text-neutral"
            )}
          >
            {evidence.direction}
          </Badge>
          <Badge variant="outline" className={cn("text-xs", getStrengthStyles())}>
            {evidence.strength}
          </Badge>
          <span className="text-xs text-muted-foreground font-mono ml-auto">
            {(evidence.sourceCredibility * 100).toFixed(0)}% cred
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
