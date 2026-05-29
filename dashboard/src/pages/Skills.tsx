import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Search,
  ChevronDown,
  ChevronUp,
  Play,
  Layers,
  BookOpen,
  Brain,
} from "lucide-react"
import { getSkills, searchSkills, runEvolution } from "@/lib/api"
import type { Skill } from "@/lib/mock-data"

export function Skills() {
  const [skills, setSkills] = useState<Skill[]>([])
  const [filtered, setFiltered] = useState<Skill[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [evolvingId, setEvolvingId] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      try {
        const s = await getSkills()
        setSkills(s)
        setFiltered(s)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  useEffect(() => {
    if (!searchQuery.trim()) {
      setFiltered(skills)
      return
    }
    const q = searchQuery.toLowerCase()
    setFiltered(
      skills.filter(
        (s) =>
          s.name.toLowerCase().includes(q) ||
          s.description.toLowerCase().includes(q)
      )
    )
  }, [searchQuery, skills])

  const handleEvolution = async (skillId: string) => {
    setEvolvingId(skillId)
    try {
      await runEvolution(skillId)
    } finally {
      setEvolvingId(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading skills...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Search and Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search skills..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <Badge variant="outline" className="text-sm">
          {filtered.length} skills
        </Badge>
      </div>

      {/* Skills Table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-8"></TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Q-Value</TableHead>
                <TableHead>Success Rate</TableHead>
                <TableHead>Usage</TableHead>
                <TableHead>Health</TableHead>
                <TableHead className="w-32">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((skill) => (
                <SkillRow
                  key={skill.id}
                  skill={skill}
                  expanded={expandedId === skill.id}
                  onToggle={() =>
                    setExpandedId(expandedId === skill.id ? null : skill.id)
                  }
                  onEvolve={() => handleEvolution(skill.id)}
                  evolving={evolvingId === skill.id}
                />
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}

function SkillRow({
  skill,
  expanded,
  onToggle,
  onEvolve,
  evolving,
}: {
  skill: Skill
  expanded: boolean
  onToggle: () => void
  onEvolve: () => void
  evolving: boolean
}) {
  return (
    <>
      <TableRow
        className="cursor-pointer"
        onClick={onToggle}
      >
        <TableCell>
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </TableCell>
        <TableCell>
          <div>
            <span className="font-medium">{skill.name}</span>
            <p className="text-xs text-muted-foreground mt-0.5 max-w-xs truncate">
              {skill.description}
            </p>
          </div>
        </TableCell>
        <TableCell>
          <div className="flex items-center gap-2">
            <Progress
              value={skill.q_value * 100}
              className="w-24"
              indicatorClassName={
                skill.q_value > 0.8
                  ? "bg-green-400"
                  : skill.q_value > 0.6
                  ? "bg-yellow-400"
                  : "bg-red-400"
              }
            />
            <span className="text-sm text-muted-foreground w-10">
              {skill.q_value.toFixed(2)}
            </span>
          </div>
        </TableCell>
        <TableCell>
          <span
            className={
              skill.success_rate > 0.8
                ? "text-green-400"
                : skill.success_rate > 0.6
                ? "text-yellow-400"
                : "text-red-400"
            }
          >
            {(skill.success_rate * 100).toFixed(1)}%
          </span>
        </TableCell>
        <TableCell className="text-muted-foreground">
          {skill.usage_count.toLocaleString()}
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
        <TableCell>
          <Button
            size="sm"
            variant="outline"
            onClick={(e) => {
              e.stopPropagation()
              onEvolve()
            }}
            disabled={evolving}
            className="gap-1"
          >
            <Play className="h-3 w-3" />
            {evolving ? "Running..." : "Evolve"}
          </Button>
        </TableCell>
      </TableRow>

      {expanded && (
        <TableRow>
          <TableCell colSpan={7} className="bg-muted/30">
            <div className="p-4 space-y-4">
              <p className="text-sm text-muted-foreground">{skill.description}</p>

              {skill.dependencies.length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                    Dependencies
                  </h4>
                  <div className="flex gap-2 flex-wrap">
                    {skill.dependencies.map((dep) => (
                      <Badge key={dep} variant="outline">
                        {dep}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <TierCard
                  icon={<BookOpen className="h-4 w-4 text-blue-400" />}
                  tier="Tier 1"
                  content={skill.tier1}
                />
                <TierCard
                  icon={<Layers className="h-4 w-4 text-purple-400" />}
                  tier="Tier 2"
                  content={skill.tier2}
                />
                <TierCard
                  icon={<Brain className="h-4 w-4 text-cyan-400" />}
                  tier="Tier 3"
                  content={skill.tier3}
                />
              </div>

              <div className="flex gap-6 text-xs text-muted-foreground">
                <span>Created: {new Date(skill.created_at).toLocaleDateString()}</span>
                <span>Updated: {new Date(skill.updated_at).toLocaleDateString()}</span>
              </div>
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

function TierCard({
  icon,
  tier,
  content,
}: {
  icon: React.ReactNode
  tier: string
  content: string
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-2">
          {icon}
          <span className="text-sm font-medium">{tier}</span>
        </div>
        <p className="text-xs text-muted-foreground">{content}</p>
      </CardContent>
    </Card>
  )
}
