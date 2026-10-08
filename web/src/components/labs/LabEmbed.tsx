"use client";
import { REGISTRY_A } from "./registry-a";
import { REGISTRY_B } from "./registry-b";
import { REGISTRY_C } from "./registry-c";
import { REGISTRY_D } from "./registry-d";
import { REGISTRY_E } from "./registry-e";

// Each group of labs owns its own registry file so they can be developed independently.
export const LAB_COMPONENTS = { ...REGISTRY_A, ...REGISTRY_B, ...REGISTRY_C, ...REGISTRY_D, ...REGISTRY_E };

export function LabEmbed({ id }: { id: string }) {
  const Lab = LAB_COMPONENTS[id];
  if (!Lab) {
    if (process.env.NODE_ENV !== "production") return <p className="text-bad">Unknown lab: {id}</p>;
    return null;
  }
  return <Lab />;
}
