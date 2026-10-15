import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";
import type { GeometryObject, GeomView, ObjectView } from "../types";
import { geomColor } from "./geomColor";

interface Props {
  geometry: GeometryObject[];
  objects: Record<string, ObjectView>;
  selected: string | null;
  onSelect: (name: string) => void;
}

// MuJoCo is Z-up and puts capsule/cylinder axes on local Z; three.js builds
// them on Y, so those two get a quarter turn inside their group.
const AXIS_FIX: [number, number, number] = [Math.PI / 2, 0, 0];

function MeshGeometry({ geom }: { geom: GeomView }) {
  const buffer = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(geom.vertices ?? [], 3));
    g.setIndex(geom.faces ?? []);
    g.computeVertexNormals();
    return g;
  }, [geom.vertices, geom.faces]);
  return <primitive object={buffer} attach="geometry" />;
}

function Shape({ geom }: { geom: GeomView }) {
  const [a, b, c] = geom.size;
  switch (geom.type) {
    case "box": return <boxGeometry args={[2 * a, 2 * b, 2 * c]} />;
    case "sphere": return <sphereGeometry args={[a, 32, 16]} />;
    case "ellipsoid": return <sphereGeometry args={[1, 32, 16]} />;
    case "cylinder": return <cylinderGeometry args={[a, a, 2 * b, 32]} />;
    case "capsule": return <capsuleGeometry args={[a, 2 * b, 4, 16]} />;
    case "plane": return <planeGeometry args={[a || 20, b || 20]} />;
    case "mesh": return <MeshGeometry geom={geom} />;
    default: return null;
  }
}

function Geom({ geom, color, emissive, onClick }: { geom: GeomView; color: string; emissive: boolean; onClick?: () => void }) {
  const quaternion = useMemo(
    () => new THREE.Quaternion(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]),
    [geom.quat],
  );
  const rotated = geom.type === "cylinder" || geom.type === "capsule";
  const scale: [number, number, number] = geom.type === "ellipsoid" ? [geom.size[0], geom.size[1], geom.size[2]] : [1, 1, 1];
  return (
    <group position={geom.pos as [number, number, number]} quaternion={quaternion} onClick={onClick && ((e) => { e.stopPropagation(); onClick(); })}>
      <mesh rotation={rotated ? AXIS_FIX : undefined} scale={scale} receiveShadow castShadow>
        <Shape geom={geom} />
        <meshStandardMaterial color={color} emissive={emissive ? color : "#000000"} emissiveIntensity={emissive ? 0.35 : 0} side={geom.type === "plane" ? THREE.DoubleSide : THREE.FrontSide} />
      </mesh>
    </group>
  );
}

export function SceneViewport({ geometry, objects, selected, onSelect }: Props) {
  return (
    <div className="viewport">
      <Canvas camera={{ position: [2.4, -2.4, 1.9], up: [0, 0, 1], fov: 45, near: 0.01, far: 2000 }} shadows>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, -2, 6]} intensity={1.2} castShadow />
        {geometry.map((obj) =>
          obj.geoms.map((geom, i) => {
            const view = obj.name === "world" ? undefined : objects[obj.name];
            const isSelected = obj.name === selected;
            return (
              <Geom
                key={`${obj.name}:${i}`}
                geom={geom}
                color={geomColor(view, geom.rgba, isSelected)}
                emissive={isSelected}
                onClick={view ? () => onSelect(obj.name) : undefined}
              />
            );
          }),
        )}
        <OrbitControls target={[0, 0, 0.6]} makeDefault />
      </Canvas>
      {geometry.length === 0 && <div className="viewport-empty">Load a scene to see it here</div>}
    </div>
  );
}
