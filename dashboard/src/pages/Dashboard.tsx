import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Zap, Target, TrendingUp, Activity } from "lucide-react"
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts"
import { getMetrics, getHealth, getSkills, getEvolutionHistory } from "@/lib/api"
import type { MetricsData, HealthData, Skill, EvolutionEvent } from "@/lib/mock-data"

export function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null)
  const [health, setHealth] = useState<HealthData | null>(null)
  const [skills, setSkills] = useState<Skill[]>([])
  const [events, setEvents] = useState<EvolutionEvent[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      try {
        const [m, h, s, e] = await Promise.all([
          getMetrics(),
          getHealth(),
          getSkills(),
          getEvolutionHistory(),
        ])
        setMetrics(m)
        setHealth(h)
        setSkills(s)
        setEvents(e)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading dashboard...</div>
      </div>
    )
  }

  const healthPieData = health
    ? [
        { name: "Healthy", value: health.healthy, color: "#4ade80" },
        { name: "Warning", value: health.warning, color: "#facc15" },
        { name: "Critical", value: health.critical, color: "#f87171" },
      ]
    : []

  const topSkills = [...skills].sort((a, b) => b.q_value - a.q_value).slice(0, 5)

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          icon={<Zap className="h-5 w-5 text-cyan-400" />}
          title="Total Skills"
          value={metrics?.total_skills ?? 0}
        />
        <KpiCard
          icon={<Target className="h-5 w-5 text-purple-400" />}
          title="Avg Q-Value"
          value={(metrics?.avg_q_value ?? 0).toFixed(3)}
        />
        <KpiCard
          icon={<TrendingUp className="h-5 w-5 text-green-400" />}
          title="Avg Success Rate"
          value={`${((metrics?.avg_success_rate ?? 0) * 100).toFixed(1)}%`}
        />
        <KpiCard
          icon={<Activity className="h-5 w-5 text-orange-400" />}
          title="Total Outcomes"
          value={(metrics?.total_outcomes ?? 0).toLocaleString()}
        />
      </div>

      {/* Charts Row */}
      <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
        {/* Health Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Skill Health Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={healthPieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={4}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {healthPieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "#18181b",
                      border: "1px solid #27272a",
                      borderRadius: "8px",
                      color: "#fafafa",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Token Usage Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Token Usage Trend (7 days)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metrics?.token_usage_trend ?? []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(d: string) => d.slice(5)}
                    stroke="#71717a"
                    fontSize={12}
                  />
                  <YAxis stroke="#71717a" fontSize={12} tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{
                      background: "#18181b",
                      border: "1px solid #27272a",
                      borderRadius: "8px",
                      color: "#fafafa",
                    }}
                    labelFormatter={(d) => String(d)}
                  />
                  <Line
                    type="monotone"
                    dataKey="tokens"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    dot={{ fill: "#22d3ee", r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
        {/* Top 5 Skills */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Top 5 Skills by Q-Value</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Skill</TableHead>
                  <TableHead>Q-Value</TableHead>
                  <TableHead>Health</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {topSkills.map((skill) => (
                  <TableRow key={skill.id}>
                    <TableCell className="font-medium">{skill.name}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Progress
                          value={skill.q_value * 100}
                          className="w-20"
                          indicatorClassName={
                            skill.q_value > 0.8
                              ? "bg-green-400"
                              : skill.q_value > 0.6
                              ? "bg-yellow-400"
                              : "bg-red-400"
                          }
                        />
                        <span className="text-xs text-muted-foreground">
                          {skill.q_value.toFixed(2)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          skill.health === "HEALTHY"
                            ? "healthy"
                            : skill.health === "WARNING"
                            ? "warning"
                            : "critical"
                        }
                      >
                        {skill.health}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Recent Evolution Events */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Recent Evolution Events</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {events.slice(0, 5).map((event) => (
                <div
                  key={event.id}
                  className="flex items-start gap-3 rounded-lg border border-border p-3"
                >
                  <div
                    className={`mt-1 h-2 w-2 rounded-full shrink-0 ${
                      event.success ? "bg-green-400" : "bg-red-400"
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{event.skill_name}</span>
                      <Badge variant="outline" className="text-xs">
                        {event.action_type}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 truncate">
                      {event.details}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {new Date(event.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function KpiCard({
  icon,
  title,
  value,
}: {
  icon: React.ReactNode
  title: string
  value: string | number
}) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold mt-1">{value}</p>
          </div>
          <div className="rounded-lg bg-muted p-3">{icon}</div>
        </div>
      </CardContent>
    </Card>
  )
}
