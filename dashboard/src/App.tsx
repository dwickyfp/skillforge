import { BrowserRouter, Routes, Route } from "react-router-dom"
import { Sidebar } from "./components/layout/Sidebar"
import { Dashboard } from "./pages/Dashboard"
import { Skills } from "./pages/Skills"
import { Evolution } from "./pages/Evolution"
import { Graph } from "./pages/Graph"
import { Settings } from "./pages/Settings"
import { Header } from "./components/layout/Header"
import { useLocation } from "react-router-dom"
import { TooltipProvider } from "./components/ui/tooltip"

function pageMeta(pathname: string): { title: string; showSearch: boolean } {
  switch (pathname) {
    case "/":
      return { title: "Dashboard", showSearch: false }
    case "/skills":
      return { title: "Skills", showSearch: true }
    case "/evolution":
      return { title: "Evolution", showSearch: false }
    case "/graph":
      return { title: "Dependency Graph", showSearch: false }
    case "/settings":
      return { title: "Settings", showSearch: false }
    default:
      return { title: "SkillForge", showSearch: false }
  }
}

function AppLayout() {
  const location = useLocation()
  const { title, showSearch } = pageMeta(location.pathname)

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <div className="flex-1 md:ml-64">
        <Header title={title} showSearch={showSearch} />
        <main className="p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/skills" element={<Skills />} />
            <Route path="/evolution" element={<Evolution />} />
            <Route path="/graph" element={<Graph />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

function App() {
  return (
    <TooltipProvider>
      <BrowserRouter>
        <AppLayout />
      </BrowserRouter>
    </TooltipProvider>
  )
}

export default App
