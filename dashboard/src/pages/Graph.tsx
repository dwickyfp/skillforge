import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { GitBranch, Info } from "lucide-react"
import { getGraph } from "@/lib/api"
import type { GraphData, GraphNode } from "@/lib/mock-data"

export function Graph() {
  const [data, setData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [hoverNode, setHoverNode] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      try {
        const g = await getGraph()
        setData(g)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading graph...</div>
      </div>
    )
  }

  if (!data) return null

  const nodeMap = new Map(data.nodes.map((n) => [n.id, n]))

  const getQColor = (q: number) => {
    if (q > 0.85) return "#4ade80"
    if (q > 0.7) return "#facc15"
    if (q > 0.6) return "#fb923c"
    return "#f87171"
  }

  const getConnectedIds = (nodeId: string) => {
    const ids = new Set<string>()
    data.edges.forEach((e) => {
      if (e.source === nodeId) ids.add(e.target)
      if (e.target === nodeId) ids.add(e.source)
    })
    return ids
  }

  const connectedIds = hoverNode ? getConnectedIds(hoverNode) : new Set<string>()

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <GitBranch className="h-5 w-5 text-cyan-400" />
        <h2 className="text-lg font-medium">Skill Dependency Graph</h2>
        <Badge variant="outline">{data.nodes.length} nodes</Badge>
        <Badge variant="outline">{data.edges.length} edges</Badge>
      </div>

      <div className="grid gap-6 grid-cols-1 lg:grid-cols-4">
        {/* Graph Visualization */}
        <Card className="lg:col-span-3">
          <CardContent className="p-6">
            <svg
              viewBox="0 0 800 550"
              className="w-full h-auto"
              style={{ minHeight: "400px" }}
            >
              {/* Edges */}
              {data.edges.map((edge, i) => {
                const source = nodeMap.get(edge.source)
                const target = nodeMap.get(edge.target)
                if (!source || !target) return null

                const isHighlighted =
                  hoverNode === edge.source || hoverNode === edge.target

                return (
                  <line
                    key={i}
                    x1={source.x}
                    y1={source.y}
                    x2={target.x}
                    y2={target.y}
                    stroke={isHighlighted ? "#22d3ee" : "#27272a"}
                    strokeWidth={isHighlighted ? 2.5 : 1.5}
                    strokeDasharray={isHighlighted ? "" : "4 2"}
                    opacity={hoverNode && !isHighlighted ? 0.2 : 1}
                  />
                )
              })}

              {/* Nodes */}
              {data.nodes.map((node) => {
                const isHovered = hoverNode === node.id
                const isConnected = connectedIds.has(node.id)
                const isDimmed = hoverNode && !isHovered && !isConnected
                const radius = 20 + node.q_value * 15

                return (
                  <g
                    key={node.id}
                    onMouseEnter={() => setHoverNode(node.id)}
                    onMouseLeave={() => setHoverNode(null)}
                    onClick={() =>
                      setSelectedNode(
                        selectedNode?.id === node.id ? null : node
                      )
                    }
                    className="cursor-pointer"
                  >
                    {/* Glow effect */}
                    {(isHovered || isConnected) && (
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={radius + 8}
                        fill="none"
                        stroke={getQColor(node.q_value)}
                        strokeWidth={1}
                        opacity={0.3}
                      />
                    )}

                    {/* Node circle */}
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={radius}
                      fill={getQColor(node.q_value)}
                      opacity={isDimmed ? 0.15 : 0.2}
                      stroke={getQColor(node.q_value)}
                      strokeWidth={isHovered ? 3 : 1.5}
                    />

                    {/* Inner filled circle */}
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={radius * 0.4}
                      fill={getQColor(node.q_value)}
                      opacity={isDimmed ? 0.2 : 0.8}
                    />

                    {/* Label */}
                    <text
                      x={node.x}
                      y={node.y + radius + 18}
                      textAnchor="middle"
                      fill={isDimmed ? "#52525b" : "#fafafa"}
                      fontSize={11}
                      fontWeight={500}
                    >
                      {node.name.replace("_", " ")}
                    </text>

                    {/* Q-value on hover */}
                    {isHovered && (
                      <text
                        x={node.x}
                        y={node.y - radius - 10}
                        textAnchor="middle"
                        fill={getQColor(node.q_value)}
                        fontSize={12}
                        fontWeight={600}
                      >
                        Q: {node.q_value.toFixed(2)}
                      </text>
                    )}
                  </g>
                )
              })}
            </svg>
          </CardContent>
        </Card>

        {/* Info Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Info className="h-4 w-4" />
                Legend
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-green-400" />
                <span className="text-sm text-muted-foreground">Q &gt; 0.85</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-yellow-400" />
                <span className="text-sm text-muted-foreground">Q &gt; 0.70</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-orange-400" />
                <span className="text-sm text-muted-foreground">Q &gt; 0.60</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-red-400" />
                <span className="text-sm text-muted-foreground">Q &lt; 0.60</span>
              </div>
              <div className="text-xs text-muted-foreground pt-2 border-t border-border">
                Node size reflects Q-value. Lines show dependencies.
              </div>
            </CardContent>
          </Card>

          {selectedNode && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">{selectedNode.name.replace("_", " ")}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Q-Value</span>
                  <span className="font-mono">{selectedNode.q_value.toFixed(3)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Dependencies</span>
                  <span className="font-mono">
                    {data.edges.filter((e) => e.source === selectedNode.id).length}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Dependents</span>
                  <span className="font-mono">
                    {data.edges.filter((e) => e.target === selectedNode.id).length}
                  </span>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
