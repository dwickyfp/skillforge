import { useState } from "react"
import { NavLink } from "react-router-dom"
import {
  LayoutDashboard,
  Zap,
  GitBranch,
  Settings,
  Flame,
  Menu,
  X,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/skills", icon: Zap, label: "Skills" },
  { to: "/evolution", icon: Flame, label: "Evolution" },
  { to: "/graph", icon: GitBranch, label: "Graph" },
  { to: "/settings", icon: Settings, label: "Settings" },
]

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <>
      {/* Mobile toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-50 md:hidden"
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {/* Overlay for mobile */}
      {collapsed && (
        <div
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          onClick={() => setCollapsed(false)}
        />
      )}

      <aside
        className={cn(
          "fixed left-0 top-0 z-40 h-screen w-64 border-r border-border bg-sidebar transition-transform md:translate-x-0",
          collapsed ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        )}
      >
        <div className="flex h-16 items-center gap-2 border-b border-border px-6">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500">
            <Zap className="h-5 w-5 text-black" />
          </div>
          <span className="text-lg font-bold">SkillForge</span>
        </div>

        <nav className="space-y-1 p-4">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              onClick={() => setCollapsed(false)}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground"
                )
              }
            >
              <item.icon className="h-5 w-5 shrink-0" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="absolute bottom-4 left-4 right-4">
          <div className="rounded-lg bg-muted p-4">
            <p className="text-xs text-muted-foreground">SkillForge v0.1.0</p>
            <p className="text-xs text-muted-foreground mt-1">Self-evolving AI Skills</p>
          </div>
        </div>
      </aside>
    </>
  )
}
