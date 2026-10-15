import { useEffect, useState } from "react";
import { api } from "../api";

interface Props {
  busy: boolean;
  onScene: (sceneId: string) => void;
  onError: (message: string) => void;
}

export function SceneInput({ busy, onScene, onError }: Props) {
  const [samples, setSamples] = useState<{ id: string; label: string }[]>([]);

  useEffect(() => {
    api.samples().then(setSamples).catch((e) => onError(e.message));
  }, [onError]);

  const pick = (promise: Promise<{ scene_id: string }>) =>
    promise.then((r) => onScene(r.scene_id)).catch((e) => onError(e.message));

  return (
    <div className="scene-input">
      {samples.map((s) => (
        <button key={s.id} disabled={busy} onClick={() => pick(api.createFromSample(s.id))}>
          Load sample: {s.label}
        </button>
      ))}
      <label className={busy ? "upload disabled" : "upload"}>
        Upload MJCF (.xml, ≤ 1 MB)
        <input
          type="file"
          accept=".xml,text/xml"
          disabled={busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) pick(api.upload(file));
            e.target.value = "";
          }}
        />
      </label>
    </div>
  );
}
