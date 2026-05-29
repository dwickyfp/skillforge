import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Flame, Play, Clock, CheckCircle, XCircle, BarChart3 } from "lucide-react"
import { getEvolutionHistory, runEvolution } from "@/lib/api"
import type { EvolutionEvent } from "@/lib/mock-data"

export function Evolution() {
  const [events, setEvents] = useState<EvolutionEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      try {
        const e = await getEvolutionHistory()
        setEvents(e)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const handleRunEvolution = async () => {
    setRunning(true)
    setToast(null)
    try {
      const res = await runEvolution()
      setToast(res.message)
      // Reload events
      const e = await getEvolutionHistory()
      setEvents(e)
    } finally {
      setRunning(false)
    }
  }

  const successCount = events.filter((e) => e.success).length
  const successRate = events.length > 0 ? successCount / events.length : 0
  const avgDuration =
    events.length > 0
      ? events.reduce((acc, e) => acc + e.duration_ms, 0) / events.length
      : 0

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading evolution history...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Toast notification */}
      {toast && (
        <div className="fixed top-20 right-6 z-50 rounded-lg border border-border bg-card p-4 shadow-lg animate-in slide-in-from-right">
          <p className="text-sm">{toast}</p>
        </div>
      )}

      {/* Stats Summary */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-3">
        <Card>
          <CardContent className="p-6 flex items-center gap-4">
            <div className="rounded-lg bg-cyan-500/10 p-3">
              <BarChart3 className="h-6 w-6 text-cyan-400" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Evolutions</p>
              <p className="text-2xl font-bold">{events.length}</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 flex items-center gap-4">
            <div className="rounded-lg bg-green-500/10 p-3">
              <CheckCircle className="h-6 w-6 text-green-400" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Success Rate</p>
              <p className="text-2xl font-bold">{(successRate * 100).toFixed(1)}%</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 flex items-center gap-4">
            <div className="rounded-lg bg-purple-500/10 p-3">
              <Clock className="h-6 w-6 text-purple-400" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Avg Duration</p>
              <p className="text-2xl font-bold">{(avgDuration / 1000).toFixed(1)}s</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Action Button */}
      <div className="flex items-center gap-4">
        <Button onClick={handleRunEvolution} disabled={running} className="gap-2">
          <Play className="h-4 w-4" />
          {running ? "Running Evolution..." : "Run Full Evolution Loop"}
        </Button>
        <span className="text-sm text-muted-foreground">
          Triggers evolution across all skills
        </span>
      </div>

      {/* Evolution Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Flame className="h-5 w-5 text-orange-400" />
            Evolution Timeline
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-6 top-0 bottom-0 w-px bg-border" />

            <div className="space-y-4">
              {events.map((event, index) => (
                <EvolutionCard key={event.id} event={event} index={index} />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function EvolutionCard({ event, index }: { event: EvolutionEvent; index: number }) {
  return (
    <div className="relative flex gap-4 pl-4">
      {/* Timeline dot */}
      <div
        className={`absolute left-[18px] top-4 h-4 w-4 rounded-full border-2 border-background z-10 ${
          event.success ? "bg-green-400" : "bg-red-400"
        }`}
      />

      {/* Content */}
      <div className="flex-1 ml-8">
        <div className="rounded-lg border border-border p-4 hover:bg-muted/30 transition-colors">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <span className="font-medium">{event.skill_name}</span>
              <Badge
                variant={
                  event.action_type === "PROMOTE_TIER"
                    ? "healthy"
                    : event.action_type === "DEMOTE_TIER"
                    ? "critical"
                    : event.action_type === "PRUNE"
                    ? "destructive"
                    : "secondary"
                }
              >
                {event.action_type}
              </Badge>
              {event.success ? (
                <CheckCircle className="h-4 w-4 text-green-400" />
              ) : (
                <XCircle className="h-4 w-4 text-red-400" />
              )}
            </div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {(event.duration_ms / 1000).toFixed(1)}s
              </span>
              <span>{new Date(event.timestamp).toLocaleString()}</span>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mt-2">{event.details}</p>

          {index < 7 && <Separator className="mt-4" />}
        </div>
      </div>
    </div>
  )
}
