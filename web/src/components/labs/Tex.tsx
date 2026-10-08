"use client";
import katex from "katex";
import { useMemo } from "react";

/** Render a LaTeX string. `display` for block equations. */
export function Tex({ children, display = false }: { children: string; display?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: display, throwOnError: false, strict: false }), [children, display]);
  return <span className={display ? "block overflow-x-auto" : undefined} dangerouslySetInnerHTML={{ __html: html }} />;
}
