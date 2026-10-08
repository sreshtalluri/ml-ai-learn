"use client";
import { LAB_COMPONENTS } from "./registry";

export function LabEmbed({ id }: { id: string }) {
  const Lab = LAB_COMPONENTS[id];
  if (!Lab) return process.env.NODE_ENV !== "production" ? <p className="text-bad">Unknown lab: {id}</p> : null;
  return <Lab />;
}
