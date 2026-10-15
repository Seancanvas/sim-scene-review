import { describe, expect, it } from "vitest";
import { geomColor } from "./geomColor";
import type { ObjectView } from "../types";

const base = (over: Partial<ObjectView>): ObjectView => ({
  name: "o", movable: true, status: "ok", decision: "pending", findings: [], ...over,
});
const finding = (severity: "error" | "warning") => ({ object: "o", rule: "r", severity, message: "", fix: null });

describe("geomColor", () => {
  it("uses the geom's own colour when there is nothing to say", () => {
    expect(geomColor(base({}), [0.2, 0.4, 0.6, 1], false)).toBe("#336699");
  });
  it("paints errors red, warnings amber, approvals green, rejections grey", () => {
    expect(geomColor(base({ findings: [finding("error")] }), [1, 1, 1, 1], false)).toBe("#e5484d");
    expect(geomColor(base({ findings: [finding("warning")] }), [1, 1, 1, 1], false)).toBe("#f5a524");
    expect(geomColor(base({ decision: "approved", findings: [finding("error")] }), [1, 1, 1, 1], false)).toBe("#30a46c");
    expect(geomColor(base({ decision: "rejected" }), [1, 1, 1, 1], false)).toBe("#8b8d98");
  });
  it("world geoms keep their colour", () => {
    expect(geomColor(undefined, [0.85, 0.85, 0.85, 1], false)).toBe("#d9d9d9");
  });
});
