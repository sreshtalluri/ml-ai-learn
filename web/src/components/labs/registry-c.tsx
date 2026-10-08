"use client";
import type { ComponentType } from "react";

// Lab group C. Map lab id (see src/lib/labs.ts) to a component, lazily loaded with next/dynamic.
export const REGISTRY_C: Record<string, ComponentType> = {};
