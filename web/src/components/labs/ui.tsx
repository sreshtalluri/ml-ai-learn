"use client";
// Shared building blocks for every lab, so all labs look and behave the same:
// title, controls with labels, presets, reset, an explanation of what changed, and an interpretation.
import { useEffect, type ReactNode } from "react";
import { ArrowCounterClockwise } from "@phosphor-icons/react";
import { actions } from "@/lib/progress";

export function LabFrame({
  id,
  title,
  subtitle,
  controls,
  presets,
  onReset,
  readout,
  interpretation,
  children,
}: {
  id: string;
  title: string;
  subtitle?: ReactNode;
  controls?: ReactNode;
  presets?: { label: string; apply: () => void }[];
  onReset?: () => void;
  readout?: ReactNode;          // live numbers (metrics, current step)
  interpretation?: ReactNode;   // 1-3 sentences on what the current state means
  children: ReactNode;          // the visualization
}) {
  useEffect(() => actions.touchLab(id), [id]);
  return (
    <section aria-label={title} className="@container not-prose my-8 rounded-xl border border-line bg-surface shadow-sm">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-4 py-3 sm:px-5">
        <div>
          <h3 className="font-semibold tracking-tight">{title}</h3>
          {subtitle && <p className="text-sm text-muted mt-0.5">{subtitle}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {presets?.map((p) => (
            <button key={p.label} type="button" onClick={p.apply} className="rounded-full border border-line px-2.5 py-1 text-xs text-muted hover:text-ink hover:border-faint">
              {p.label}
            </button>
          ))}
          {onReset && (
            <button type="button" onClick={onReset} className="inline-flex items-center gap-1 rounded-full border border-line px-2.5 py-1 text-xs text-muted hover:text-ink hover:border-faint">
              <ArrowCounterClockwise size={12} /> Reset
            </button>
          )}
        </div>
      </header>
      {/* Container query: side panel only when the lab itself is wide (labs page), stacked inside lessons. */}
      <div className="grid gap-0 @4xl:grid-cols-[minmax(0,1fr)_17rem]">
        <div className="p-3 sm:p-5 min-w-0">{children}</div>
        {(controls || readout) && (
          <aside className="border-t @4xl:border-t-0 @4xl:border-l border-line p-4 sm:p-5 grid gap-5 @xl:grid-cols-2 @4xl:grid-cols-1 @4xl:content-start text-sm">
            {controls && <div className="space-y-4">{controls}</div>}
            {readout && <div className="space-y-1.5 font-mono text-[0.8rem]">{readout}</div>}
          </aside>
        )}
      </div>
      {interpretation && (
        <footer className="border-t border-line px-4 py-3 sm:px-5 text-sm text-muted leading-relaxed" aria-live="polite">
          {interpretation}
        </footer>
      )}
    </section>
  );
}

export function Slider({
  label, value, min, max, step = 1, onChange, format = (v) => String(v), hint,
}: {
  label: string; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; format?: (v: number) => string; hint?: string;
}) {
  return (
    <label className="block">
      <span className="flex justify-between gap-2 text-[0.8rem]">
        <span className="text-muted">{label}</span>
        <span className="font-mono text-ink">{format(value)}</span>
      </span>
      <input type="range" className="w-full mt-1" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <span className="block text-xs text-faint mt-0.5">{hint}</span>}
    </label>
  );
}

export function Segmented<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <fieldset>
      <legend className="text-[0.8rem] text-muted mb-1">{label}</legend>
      <div className="flex flex-wrap gap-1 rounded-lg bg-surface-2 p-1">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            aria-pressed={value === o.value}
            onClick={() => onChange(o.value)}
            className={`flex-1 rounded-md px-2 py-1 text-xs whitespace-nowrap ${value === o.value ? "bg-surface text-ink shadow-sm font-medium" : "text-muted hover:text-ink"}`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-[0.8rem] text-muted cursor-pointer">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-[var(--accent)]" />
      {label}
    </label>
  );
}

export function Button({ children, onClick, primary, disabled }: { children: ReactNode; onClick: () => void; primary?: boolean; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium active:scale-[0.98] disabled:opacity-40 ${primary ? "bg-accent text-white dark:text-zinc-950" : "border border-line text-ink hover:bg-surface-2"}`}
    >
      {children}
    </button>
  );
}

/** Label/value row for the readout panel. */
export function Stat({ label, value, color }: { label: string; value: ReactNode; color?: string }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-muted">{label}</span>
      <span style={color ? { color } : undefined} className="text-ink">{value}</span>
    </div>
  );
}

// ---------- SVG plotting ----------

export interface Scale { (v: number): number; invert: (px: number) => number; domain: [number, number]; range: [number, number] }

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain, [r0, r1] = range;
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)) as Scale;
  f.invert = (px: number) => d0 + ((px - r0) / (r1 - r0)) * (d1 - d0);
  f.domain = domain;
  f.range = range;
  return f;
}

export const ticks = (d: [number, number], count = 5) => {
  const raw = (d[1] - d[0]) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(d[0] / step) * step; v <= d[1] + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
};

/** Plot area with axes, gridlines and labels. Children draw in data space via the x/y scales. */
export function Plot({
  width = 560, height = 360, x, y, xLabel, yLabel, margin = { t: 12, r: 12, b: 40, l: 48 }, children, svgProps, title,
}: {
  width?: number; height?: number;
  x: [number, number]; y: [number, number];
  xLabel?: string; yLabel?: string; title: string;
  margin?: { t: number; r: number; b: number; l: number };
  children: (s: { sx: Scale; sy: Scale; width: number; height: number }) => ReactNode;
  svgProps?: React.SVGProps<SVGSVGElement>;
}) {
  const sx = linear(x, [margin.l, width - margin.r]);
  const sy = linear(y, [height - margin.b, margin.t]);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title} className="w-full h-auto select-none touch-none" {...svgProps}>
      <title>{title}</title>
      {ticks(x).map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} x2={sx(t)} y1={margin.t} y2={height - margin.b} stroke="var(--grid)" />
          <text x={sx(t)} y={height - margin.b + 16} textAnchor="middle" fontSize="11" fill="var(--faint)">{t}</text>
        </g>
      ))}
      {ticks(y).map((t) => (
        <g key={`y${t}`}>
          <line x1={margin.l} x2={width - margin.r} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
          <text x={margin.l - 8} y={sy(t) + 4} textAnchor="end" fontSize="11" fill="var(--faint)">{t}</text>
        </g>
      ))}
      {xLabel && <text x={(margin.l + width - margin.r) / 2} y={height - 6} textAnchor="middle" fontSize="12" fill="var(--muted)">{xLabel}</text>}
      {yLabel && <text transform={`translate(13 ${(margin.t + height - margin.b) / 2}) rotate(-90)`} textAnchor="middle" fontSize="12" fill="var(--muted)">{yLabel}</text>}
      {children({ sx, sy, width, height })}
    </svg>
  );
}

/** Convert a pointer event to SVG viewBox coordinates. */
export function svgPoint(e: React.PointerEvent, svg: SVGSVGElement) {
  const pt = svg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  const m = svg.getScreenCTM();
  return m ? pt.matrixTransform(m.inverse()) : pt;
}

/** Small legend row. */
export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted mt-2">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <svg width="16" height="8" aria-hidden><line x1="0" x2="16" y1="4" y2="4" stroke={i.color} strokeWidth="3" strokeDasharray={i.dashed ? "4 3" : undefined} /></svg>
          {i.label}
        </span>
      ))}
    </div>
  );
}

/** Inline KaTeX rendering for client components. */
export { Tex } from "./Tex";
