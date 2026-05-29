import { Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import { ThemeToggle } from "./ThemeToggle"

interface HeaderProps {
  title: string
  searchQuery?: string
  onSearchChange?: (query: string) => void
  showSearch?: boolean
}

export function Header({ title, searchQuery, onSearchChange, showSearch = true }: HeaderProps) {
  return (
    <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6">
      <div className="flex items-center gap-4 pl-10 md:pl-0">
        <h1 className="text-xl font-semibold">{title}</h1>
      </div>

      <div className="flex items-center gap-4">
        {showSearch && (
          <div className="relative hidden sm:block">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search skills..."
              value={searchQuery || ""}
              onChange={(e) => onSearchChange?.(e.target.value)}
              className="w-64 pl-9"
            />
          </div>
        )}
        <ThemeToggle />
      </div>
    </header>
  )
}
