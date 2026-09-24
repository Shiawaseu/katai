import React, { useState, useMemo } from "react";
import { Search, Layers } from "lucide-react";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { DOMElement } from "@/types";

interface ActionSpaceTableProps {
  elements: DOMElement[];
  hoveredElementIndex?: string | null;
  setHoveredElementIndex?: (index: string | null) => void;
  selectedElementIndex?: string | null;
  setSelectedElementIndex?: (index: string | null) => void;
}

export const ActionSpaceTable: React.FC<ActionSpaceTableProps> = ({
  elements = [],
  hoveredElementIndex,
  setHoveredElementIndex,
  selectedElementIndex,
  setSelectedElementIndex,
}) => {
  const [search, setSearch] = useState("");
  const [filterRole, setFilterRole] = useState<string>("all");

  const filtered = useMemo(() => {
    return elements.filter((el) => {
      const matchSearch =
        search === "" ||
        el.label.toLowerCase().includes(search.toLowerCase()) ||
        el.role.toLowerCase().includes(search.toLowerCase()) ||
        el.index.toString() === search;

      const matchRole =
        filterRole === "all" ||
        (filterRole === "button" && el.role === "button") ||
        (filterRole === "link" && el.role === "link") ||
        (filterRole === "input" && ["textbox", "searchbox", "input", "spinbutton"].includes(el.role)) ||
        (filterRole === "select" && ["combobox", "select", "option"].includes(el.role));

      return matchSearch && matchRole;
    });
  }, [elements, search, filterRole]);

  const getOpBadge = (op: string) => {
    switch (op) {
      case "CLICK":
        return <Badge variant="info" className="text-[9px] px-1 py-0 font-mono">CLICK</Badge>;
      case "TYPE_TEXT":
        return <Badge variant="success" className="text-[9px] px-1 py-0 font-mono">TYPE</Badge>;
      case "SELECT":
        return <Badge variant="warning" className="text-[9px] px-1 py-0 font-mono">SELECT</Badge>;
      default:
        return <Badge variant="secondary" className="text-[9px] px-1 py-0 font-mono">{op}</Badge>;
    }
  };

  return (
    <div className="flex flex-col h-full bg-card rounded-lg border border-border overflow-hidden">
      {/* Search and Filters */}
      <div className="p-3 border-b border-border bg-muted/10 space-y-2">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Filter by label, role, index..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-8 h-8 text-xs bg-background border-border"
            />
          </div>
          <span className="text-[11px] font-mono text-muted-foreground px-2">
            {filtered.length} / {elements.length}
          </span>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1 overflow-x-auto pb-0.5">
          {["all", "button", "link", "input", "select"].map((role) => (
            <button
              key={role}
              onClick={() => setFilterRole(role)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider transition-colors border ${
                filterRole === role
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-transparent text-muted-foreground hover:text-foreground border-border hover:bg-secondary"
              }`}
            >
              {role}
            </button>
          ))}
        </div>
      </div>

      {/* Table Content */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-8 text-center text-muted-foreground">
            <Layers className="h-6 w-6 mb-2 opacity-30" />
            <p className="text-xs">No matching interactive controls found</p>
          </div>
        ) : (
          <table className="w-full text-left text-xs border-collapse">
            <thead className="sticky top-0 bg-muted/90 backdrop-blur border-b border-border text-[10px] uppercase font-semibold text-muted-foreground tracking-wider z-10">
              <tr>
                <th className="py-2 px-3 w-12 font-mono">#</th>
                <th className="py-2 px-3 w-24">Role</th>
                <th className="py-2 px-3">Label</th>
                <th className="py-2 px-3 w-28">Operations</th>
                <th className="py-2 px-3 w-32">Value</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40 font-mono">
              {filtered.map((el) => {
                const isHovered = hoveredElementIndex === el.index;
                const isSelected = selectedElementIndex === el.index;
                return (
                  <tr
                    key={el.index}
                    onMouseEnter={() => setHoveredElementIndex?.(el.index)}
                    onMouseLeave={() => setHoveredElementIndex?.(null)}
                    onClick={() => setSelectedElementIndex?.(el.index)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? "bg-secondary text-foreground"
                        : isHovered
                        ? "bg-secondary/60 text-foreground"
                        : "hover:bg-secondary/30 text-foreground/90"
                    }`}
                  >
                    <td className="py-2 px-3 font-bold text-foreground">{el.index}</td>
                    <td className="py-2 px-3 font-sans">
                      <Badge variant="secondary" className="text-[10px] px-1.5 py-0 border-border">
                        {el.role}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 font-sans max-w-xs truncate" title={el.label}>
                      {el.label || <span className="text-muted-foreground italic">(empty)</span>}
                    </td>
                    <td className="py-2 px-3">
                      <div className="flex flex-wrap gap-1">
                        {el.operations.map((op) => (
                          <React.Fragment key={op}>
                            {getOpBadge(op)}
                          </React.Fragment>
                        ))}
                      </div>
                    </td>
                    <td className="py-2 px-3 truncate max-w-[120px] text-muted-foreground text-[11px]">
                      {el.value || "-"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
