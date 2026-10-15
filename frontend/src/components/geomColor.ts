import type { ObjectView } from "../types";
import { statusClass } from "./ObjectPanel";

const STATUS_COLORS: Record<string, string> = {
  error: "#e5484d",
  warning: "#f5a524",
  approved: "#30a46c",
  rejected: "#8b8d98",
  checking: "#3e63dd",
};

const hex = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0");

export function geomColor(object: ObjectView | undefined, rgba: number[], selected: boolean): string {
  const own = `#${hex(rgba[0])}${hex(rgba[1])}${hex(rgba[2])}`;
  if (!object) return own;
  const color = STATUS_COLORS[statusClass(object)] ?? own;
  return selected && color === own ? "#3e63dd" : color;
}
