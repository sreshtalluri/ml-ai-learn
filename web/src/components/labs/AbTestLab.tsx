"use client";
import { useMemo, useState } from "react";
import { fmt, gaussian, rng } from "@/lib/ml";
import { LabFrame, Legend, Plot, Slider, Stat } from "./ui";

/** Standard normal CDF (Abramowitz and Stegun 7.1.26, |error| < 1.5e-7). */
export function normalCdf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const y = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-(x * x) / 2);
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}

/** Inverse standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9). */
export function normalQuantile(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - lo) return -normalQuantile(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Per-arm sample size for a two-sided two-proportion z-test (unpooled variance). */
export function sampleSize(p1: number, relMde: number, alpha = 0.05, power = 0.8): number {
  const p2 = p1 * (1 + relMde);
  const z = normalQuantile(1 - alpha / 2) + normalQuantile(power);
  return Math.ceil((z * z * (p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1) ** 2);
}

/** Power to detect a true relative lift with n users per arm (ignores the negligible opposite tail). */
export function powerAt(p1: number, relLift: number, n: number, alpha = 0.05): number {
  const p2 = p1 * (1 + relLift);
  const se = Math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n);
  return normalCdf(Math.abs(p2 - p1) / se - normalQuantile(1 - alpha / 2));
}

/**
 * Seeded A/A tests (no true effect). Daily conversions per arm are drawn from the normal approximation to the
 * binomial. Returns the false-positive rate with one look at the end, and the rate when you stop at the first
 * daily look with p < alpha (curve[d] = rate if you peek on days 1..d+1).
 */
export function simulateAA({ p, perArmPerDay, days, sims, alpha = 0.05, seed = 7 }: { p: number; perArmPerDay: number; days: number; sims: number; alpha?: number; seed?: number }) {
  const rand = rng(seed);
  const zCrit = normalQuantile(1 - alpha / 2);
  const mu = perArmPerDay * p, sd = Math.sqrt(perArmPerDay * p * (1 - p));
  const everSig = new Array(days).fill(0);
  let finalSig = 0;
  for (let s = 0; s < sims; s++) {
    let c = 0, t = 0, seen = false;
    for (let d = 0; d < days; d++) {
      c += mu + sd * gaussian(rand);
      t += mu + sd * gaussian(rand);
      const n = perArmPerDay * (d + 1), pool = (c + t) / (2 * n);
      const z = (t - c) / n / Math.sqrt((pool * (1 - pool) * 2) / n);
      const sig = Math.abs(z) > zCrit;
      seen ||= sig;
      if (seen) everSig[d]++;
      if (d === days - 1 && sig) finalSig++;
    }
  }
  return { fprFinal: finalSig / sims, fprPeek: everSig[days - 1] / sims, curve: everSig.map((v) => v / sims) };
}

const DEFAULTS = { base: 10, mde: 10, alpha: 0.05, power: 0.8, traffic: 3000, days: 14, seed: 7 };
const SIMS = 2000;

export default function AbTestLab() {
  const [st, setSt] = useState(DEFAULTS);
  const set = (patch: Partial<typeof DEFAULTS>) => setSt((s) => ({ ...s, ...patch }));
  const p1 = st.base / 100, mde = st.mde / 100;
  const n = sampleSize(p1, mde, st.alpha, st.power);
  const perArm = st.traffic / 2;
  const days = Math.ceil(n / perArm);
  const maxLift = Math.min(mde * 2.5, 1);
  const curve = useMemo(() => Array.from({ length: 81 }, (_, i) => {
    const l = (i / 80) * maxLift;
    return [l * 100, powerAt(p1, l, n, st.alpha)] as const;
  }), [p1, n, st.alpha, maxLift]);
  const sim = useMemo(() => simulateAA({ p: p1, perArmPerDay: perArm, days: st.days, sims: SIMS, alpha: st.alpha, seed: st.seed }), [p1, perArm, st.days, st.alpha, st.seed]);
  const path = (pts: readonly (readonly [number, number])[], sx: (v: number) => number, sy: (v: number) => number) =>
    pts.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ");

  return (
    <LabFrame
      id="ab-test"
      title="A/B test power and peeking"
      subtitle="Top: how many users a conversion-rate test needs. Bottom: 2,000 seeded A/A tests (no real difference) showing what checking the p-value every day does to your false-positive rate."
      onReset={() => setSt(DEFAULTS)}
      presets={[
        { label: "Small effect (3%)", apply: () => set({ mde: 3 }) },
        { label: "Rare event (1%)", apply: () => set({ base: 1 }) },
        { label: "Strict α = 0.01", apply: () => set({ alpha: 0.01 }) },
        { label: "New seed", apply: () => set({ seed: st.seed + 1 }) },
      ]}
      controls={
        <>
          <Slider label="baseline conversion" value={st.base} min={1} max={50} step={0.5} format={(v) => `${v}%`} onChange={(v) => set({ base: v })} />
          <Slider label="MDE (relative)" value={st.mde} min={1} max={50} step={1} format={(v) => `${v}%`} onChange={(v) => set({ mde: v })} hint={`${fmt(st.base, 1)}% → ${fmt(st.base * (1 + mde), 2)}%`} />
          <Slider label="α (two-sided)" value={st.alpha} min={0.01} max={0.1} step={0.01} format={(v) => fmt(v, 2)} onChange={(v) => set({ alpha: v })} />
          <Slider label="power (1 − β)" value={st.power} min={0.5} max={0.95} step={0.05} format={(v) => fmt(v, 2)} onChange={(v) => set({ power: v })} />
          <Slider label="eligible users / day" value={st.traffic} min={500} max={100000} step={500} format={(v) => v.toLocaleString("en-US")} onChange={(v) => set({ traffic: v })} />
          <Slider label="test length for simulation (days)" value={st.days} min={2} max={30} onChange={(v) => set({ days: v })} />
        </>
      }
      readout={
        <>
          <Stat label="n per arm" value={n.toLocaleString("en-US")} color="var(--c-blue)" />
          <Stat label="total users" value={(2 * n).toLocaleString("en-US")} />
          <Stat label="days needed" value={days} />
          <Stat label="z(1−α/2)" value={fmt(normalQuantile(1 - st.alpha / 2), 4)} />
          <Stat label="z(power)" value={fmt(normalQuantile(st.power), 4)} />
          <Stat label="FPR, one final look" value={fmt(sim.fprFinal, 3)} color="var(--c-teal)" />
          <Stat label="FPR, peek daily" value={fmt(sim.fprPeek, 3)} color="var(--c-orange)" />
        </>
      }
      interpretation={
        <>
          <p>
            To detect a {st.mde}% relative lift on a {fmt(st.base, 1)}% baseline with α = {fmt(st.alpha, 2)} and power {fmt(st.power, 2)}, you need {n.toLocaleString("en-US")} users per arm: {days} day{days === 1 ? "" : "s"} at {perArm.toLocaleString("en-US")} per arm per day{days < 7 ? ", but run at least one full week to cover weekday effects" : ""}. Halving the MDE roughly quadruples n.
          </p>
          <p className="mt-1">
            In {SIMS.toLocaleString("en-US")} A/A tests, one look at the end flags {fmt(sim.fprFinal * 100, 1)}% as significant (close to α). Stopping at the first daily p &lt; {fmt(st.alpha, 2)} over {st.days} days flags {fmt(sim.fprPeek * 100, 1)}%: every extra look is another chance for noise to cross the line.
          </p>
        </>
      }
    >
      <Plot title="Power versus true lift" x={[0, maxLift * 100]} y={[0, 1]} height={250} xLabel="true relative lift (%)" yLabel="power">
        {({ sx, sy }) => (
          <>
            <line x1={sx(0)} x2={sx(maxLift * 100)} y1={sy(st.power)} y2={sy(st.power)} stroke="var(--faint)" strokeDasharray="4 3" />
            <line x1={sx(st.mde)} x2={sx(st.mde)} y1={sy(0)} y2={sy(1)} stroke="var(--c-orange)" strokeDasharray="4 3" />
            <path d={path(curve, sx, sy)} fill="none" stroke="var(--c-blue)" strokeWidth={2.5} />
            <circle cx={sx(st.mde)} cy={sy(powerAt(p1, mde, n, st.alpha))} r={4} fill="var(--c-blue)" />
          </>
        )}
      </Plot>
      <Legend items={[{ label: `power with n = ${n.toLocaleString("en-US")} per arm`, color: "var(--c-blue)" }, { label: "MDE", color: "var(--c-orange)", dashed: true }, { label: "target power", color: "var(--faint)", dashed: true }]} />
      <div className="mt-4">
        <Plot title="False-positive rate when peeking daily" x={[1, Math.max(st.days, 2)]} y={[0, Math.max(0.3, sim.fprPeek * 1.2)]} height={230} xLabel="number of daily looks" yLabel="false-positive rate">
          {({ sx, sy }) => (
            <>
              <line x1={sx(1)} x2={sx(st.days)} y1={sy(st.alpha)} y2={sy(st.alpha)} stroke="var(--c-teal)" strokeDasharray="4 3" />
              <path d={path(sim.curve.map((v, i) => [i + 1, v] as const), sx, sy)} fill="none" stroke="var(--c-orange)" strokeWidth={2.5} />
              {sim.curve.map((v, i) => <circle key={i} cx={sx(i + 1)} cy={sy(v)} r={3} fill="var(--c-orange)" />)}
            </>
          )}
        </Plot>
        <Legend items={[{ label: "stop at first significant look", color: "var(--c-orange)" }, { label: `nominal α = ${fmt(st.alpha, 2)}`, color: "var(--c-teal)", dashed: true }]} />
      </div>
    </LabFrame>
  );
}
