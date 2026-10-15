import { useEffect, useRef } from "react";
import { EVENT_TYPES, type RunEvent } from "./types";

/** Subscribes to a run's SSE stream. EventSource reconnects on its own and
 * sends Last-Event-ID, so a dropped connection resumes where it stopped. */
export function useRunStream(runId: string | null, onEvent: (event: RunEvent, runId: string) => void) {
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(`/api/runs/${runId}/events`);
    for (const type of EVENT_TYPES) {
      source.addEventListener(type, (raw) => {
        const event = { type, data: JSON.parse((raw as MessageEvent).data) } as RunEvent;
        handler.current(event, runId);
        if (type === "run_completed" || type === "run_failed") source.close();
      });
    }
    return () => source.close();
  }, [runId]);
}
