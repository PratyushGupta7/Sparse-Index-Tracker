"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useMemo, useRef, useState } from "react";
import * as THREE from "three";

const N_TOTAL = 502;
const N_ACTIVE = 50;

function fibonacciSphere(n: number, radius = 2.4): [number, number, number][] {
  const points: [number, number, number][] = [];
  const phi = Math.PI * (Math.sqrt(5) - 1);
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = phi * i;
    const x = Math.cos(theta) * r;
    const z = Math.sin(theta) * r;
    points.push([x * radius, y * radius, z * radius]);
  }
  return points;
}

function Dots({ paused }: { paused: boolean }) {
  const groupRef = useRef<THREE.Group | null>(null);

  const { positions, activeSet } = useMemo(() => {
    const pos = fibonacciSphere(N_TOTAL);
    const set = new Set<number>();
    let seed = 12345;
    while (set.size < N_ACTIVE) {
      seed = (seed * 16807) % 2147483647;
      set.add(seed % N_TOTAL);
    }
    return { positions: pos, activeSet: set };
  }, []);

  useFrame((_, delta) => {
    if (!paused && groupRef.current) {
      groupRef.current.rotation.y += delta * 0.18;
      groupRef.current.rotation.x = Math.sin(performance.now() * 0.0002) * 0.15;
    }
  });

  const inactiveColor = new THREE.Color("#1e293b");
  const activeColor = new THREE.Color("#22C55E");

  return (
    <group ref={groupRef}>
      {positions.map((p, i) => {
        const active = activeSet.has(i);
        return (
          <mesh key={i} position={p}>
            <sphereGeometry args={[active ? 0.05 : 0.018, 8, 8]} />
            <meshStandardMaterial
              color={active ? activeColor : inactiveColor}
              emissive={active ? activeColor : new THREE.Color("#000000")}
              emissiveIntensity={active ? 0.9 : 0}
              roughness={0.3}
            />
          </mesh>
        );
      })}
    </group>
  );
}

export function HeroSphere() {
  const [paused, setPaused] = useState(false);
  return (
    <div
      className="aspect-square w-full max-w-md"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
      tabIndex={-1}
      aria-hidden="true"
    >
      <Canvas
        camera={{ position: [0, 0, 6], fov: 45 }}
        dpr={[1, 1.6]}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.4} />
        <pointLight position={[5, 5, 5]} intensity={1.2} color="#22C55E" />
        <pointLight position={[-5, -3, 5]} intensity={0.6} color="#F59E0B" />
        <Suspense fallback={null}>
          <Dots paused={paused} />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default HeroSphere;
