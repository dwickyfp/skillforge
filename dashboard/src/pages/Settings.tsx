import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Settings as SettingsIcon, Save, Download, Upload, ToggleLeft, ToggleRight, Bell, Server } from "lucide-react"

export function Settings() {
  const [apiEndpoint, setApiEndpoint] = useState("http://localhost:8742")
  const [autoEvolution, setAutoEvolution] = useState(false)
  const [alertThreshold, setAlertThreshold] = useState("0.5")
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleExport = () => {
    const config = {
      apiEndpoint,
      autoEvolution,
      alertThreshold,
    }
    const blob = new Blob([JSON.stringify(config, null, 2)], {
      type: "application/json",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "skillforge-config.json"
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleImport = () => {
    const input = document.createElement("input")
    input.type = "file"
    input.accept = ".json"
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = (ev) => {
        try {
          const config = JSON.parse(ev.target?.result as string)
          if (config.apiEndpoint) setApiEndpoint(config.apiEndpoint)
          if (config.autoEvolution !== undefined) setAutoEvolution(config.autoEvolution)
          if (config.alertThreshold) setAlertThreshold(config.alertThreshold)
        } catch {
          alert("Invalid config file")
        }
      }
      reader.readAsText(file)
    }
    input.click()
  }

  return (
    <div className="space-y-6 max-w-2xl">
      {saved && (
        <div className="fixed top-20 right-6 z-50 rounded-lg border border-border bg-card p-4 shadow-lg">
          <p className="text-sm text-green-400">Settings saved successfully!</p>
        </div>
      )}

      {/* API Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Server className="h-5 w-5 text-cyan-400" />
            API Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">API Endpoint</label>
            <Input
              value={apiEndpoint}
              onChange={(e) => setApiEndpoint(e.target.value)}
              placeholder="http://localhost:8742"
            />
            <p className="text-xs text-muted-foreground">
              The SkillForge API server address (default port: 8742)
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              Status:
            </Badge>
            <Badge variant="healthy">Connected</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Evolution Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <ToggleRight className="h-5 w-5 text-purple-400" />
            Evolution Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Auto-Evolution</p>
              <p className="text-xs text-muted-foreground">
                Automatically run evolution loops on a schedule
              </p>
            </div>
            <button
              onClick={() => setAutoEvolution(!autoEvolution)}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              {autoEvolution ? (
                <ToggleRight className="h-8 w-8 text-cyan-400" />
              ) : (
                <ToggleLeft className="h-8 w-8" />
              )}
            </button>
          </div>

          <Separator />

          <div className="space-y-2">
            <label className="text-sm font-medium">Evolution Schedule</label>
            <Input
              value={autoEvolution ? "Every 6 hours" : "Manual only"}
              disabled
              className="opacity-60"
            />
            <p className="text-xs text-muted-foreground">
              {autoEvolution
                ? "Evolution runs automatically every 6 hours"
                : "Enable auto-evolution to schedule automatic runs"}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Alert Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Bell className="h-5 w-5 text-orange-400" />
            Alert Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Critical Alert Threshold</label>
            <Input
              type="number"
              value={alertThreshold}
              onChange={(e) => setAlertThreshold(e.target.value)}
              min="0"
              max="1"
              step="0.05"
              placeholder="0.5"
            />
            <p className="text-xs text-muted-foreground">
              Skills with Q-value below this threshold are marked CRITICAL (0.0 - 1.0)
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Warning Threshold</label>
            <Input
              type="number"
              value="0.65"
              disabled
              className="opacity-60"
              placeholder="0.65"
            />
            <p className="text-xs text-muted-foreground">
              Skills with Q-value below this are marked WARNING (coming soon: configurable)
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Import/Export */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <SettingsIcon className="h-5 w-5 text-green-400" />
            Data Management
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-3">
            <Button variant="outline" onClick={handleExport} className="gap-2">
              <Download className="h-4 w-4" />
              Export Config
            </Button>
            <Button variant="outline" onClick={handleImport} className="gap-2">
              <Upload className="h-4 w-4" />
              Import Config
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Export or import dashboard configuration as JSON.
          </p>
        </CardContent>
      </Card>

      {/* Save Button */}
      <div className="flex justify-end">
        <Button onClick={handleSave} className="gap-2">
          <Save className="h-4 w-4" />
          Save Settings
        </Button>
      </div>
    </div>
  )
}
