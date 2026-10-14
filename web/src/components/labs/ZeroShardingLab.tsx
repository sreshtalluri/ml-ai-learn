"use client";
import { useState } from "react";
import { fmt } from "@/lib/ml";
import { activationBytes, paramCounts, PRESETS, type ModelCfg } from "./LoraLab";
import { LabFrame, Legend, linear, Segmented, Slider, Stat, Tex, ticks, type TourStep } from "./ui";

// Pure arithmetic, mirrors guide/code/27-training-at-scale/zero_sharding.py.
// Model shapes are the SYNTHETIC Llama-like presets shared with the LoRA lab.
export type Strategy = "ddp" | "zero1" | "zero2" | "zero3";
export type Precision = "mixed" | "fp32" | "adam8";

export const STRATEGIES: { id: Strategy; label: string }[] = [
  { id: "ddp", label: "DDP" }, { id: "zero1", label: "ZeRO-1" }, { id: "zero2", label: "ZeRO-2" }, { id: "zero3", label: "ZeRO-3 / FSDP" },
];
// bytes per parameter for weights, gradients, optimizer states
export const BYTES: Record<Precision, { w: number; g: number; o: number; act: number; label: string }> = {
  mixed: { w: 2, g: 2, o: 12, act: 1, label: "bf16 + AdamW" },   // optimizer = fp32 master 4 + m 4 + v 4
  fp32: { w: 4, g: 4, o: 8, act: 2, label: "fp32 + AdamW" },     // weights are already fp32: no extra master
  adam8: { w: 2, g: 2, o: 6, act: 1, label: "bf16 + 8-bit Adam" }, // fp32 master 4 + m 1 + v 1
};
export const GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const TOKENS = 2048; // per GPU micro-batch, gradient checkpointing on

/** Which of (optimizer, gradients, weights) a strategy shards across the N GPUs. */
export const sharded = (s: Strategy) => ({ o: s !== "ddp", g: s === "zero2" || s === "zero3", w: s === "zero3" });

export function perGpu(c: ModelCfg, s: Strategy, n: number, p: Precision) {
  const P = paramCounts(c).total, b = BYTES[p], sh = sharded(s);
  return {
    weights: (b.w * P) / (sh.w ? n : 1),
    grads: (b.g * P) / (sh.g ? n : 1),
    optimizer: (b.o * P) / (sh.o ? n : 1),
    activations: b.act * activationBytes(c, TOKENS, true),
  };
}

/** Bytes each GPU sends per step with ring collectives: all-reduce = 2(N-1)/N, reduce-scatter or all-gather = (N-1)/N. */
export function commBytes(c: ModelCfg, s: Strategy, n: number, p: Precision) {
  if (n === 1) return 0;
  const P = paramCounts(c).total, b = BYTES[p], f = (n - 1) / n;
  if (s === "ddp") return 2 * f * b.g * P;              // all-reduce gradients
  if (s === "zero3") return 2 * f * b.w * P + f * b.g * P; // all-gather weights (forward, backward) + reduce-scatter grads
  return f * b.g * P + f * b.w * P;                     // reduce-scatter grads + all-gather updated weights
}

export const total = (m: Record<string, number>) => Object.values(m).reduce((a, b) => a + b, 0);

/** Smallest GPU count from GPU_COUNTS whose per-GPU memory fits, or null. */
export function minGpus(c: ModelCfg, s: Strategy, p: Precision, capGB: number) {
  return GPU_COUNTS.find((n) => total(perGpu(c, s, n, p)) / 1e9 <= capGB) ?? null;
}

const GB = 1e9;
const PARTS = [
  { key: "weights", label: "weights", color: "var(--c-blue)", sh: "w" },
  { key: "grads", label: "gradients", color: "var(--c-orange)", sh: "g" },
  { key: "optimizer", label: "optimizer states", color: "var(--c-purple)", sh: "o" },
  { key: "activations", label: "activations", color: "var(--c-teal)", sh: null },
] as const;
const MODELS = ["1b", "7b", "13b", "70b"];
const W = 560, H = 210, M = { t: 24, r: 70, b: 36, l: 104 };
const MAX_CELLS = 8;

export default function ZeroShardingLab() {
  const [model, setModel] = useState("7b");
  const [ni, setNi] = useState(3);
  const [strategy, setStrategy] = useState<Strategy>("zero1");
  const [prec, setPrec] = useState<Precision>("mixed");
  const [cap, setCap] = useState<"24" | "48" | "80">("80");

  const cfg = PRESETS[model], n = GPU_COUNTS[ni], capGB = Number(cap);
  const P = paramCounts(cfg).total;
  const rows = STRATEGIES.map((s) => {
    const mem = perGpu(cfg, s.id, n, prec);
    return { ...s, mem, sum: total(mem), comm: commBytes(cfg, s.id, n, prec) };
  });
  const cur = rows.find((r) => r.id === strategy)!, ddp = rows[0];
  const ratio = ddp.comm ? cur.comm / ddp.comm : 1;
  const fitN = minGpus(cfg, strategy, prec, capGB);
  const xMax = Math.max(capGB * 1.15, ...rows.map((r) => r.sum / GB)) * 1.05;
  const sx = linear([0, xMax], [M.l, W - M.r]);
  const rowH = (H - M.t - M.b) / rows.length;

  const setup = (o: { model?: string; ni?: number; s?: Strategy; p?: Precision; cap?: "24" | "48" | "80" } = {}) => {
    setModel(o.model ?? "7b"); setNi(o.ni ?? 3); setStrategy(o.s ?? "ddp"); setPrec(o.p ?? "mixed"); setCap(o.cap ?? "80");
  };
  const reset = () => setup({ s: "zero1" });

  // Tour captions are computed with the same functions as the bars.
  const g = (v: number, d = 1) => fmt(v, d);
  const c7 = PRESETS["7b"], c70 = PRESETS["70b"];
  const at = (c: ModelCfg, s: Strategy, k: number) => total(perGpu(c, s, k, "mixed")) / GB;
  const r7 = (s: Strategy) => commBytes(c7, s, 8, "mixed") / commBytes(c7, "ddp", 8, "mixed");
  const fit70 = minGpus(c70, "zero3", "mixed", 80);
  const tour: TourStep[] = [
    { id: "ddp", caption: `A 7B-style model on 8 GPUs with plain data parallelism. Every GPU keeps a full copy of weights, gradients and optimizer states: ${g(at(c7, "ddp", 8))} GB each, far past the 80 GB line. Purple, the optimizer state, is the biggest block.`, apply: () => setup() },
    { id: "zero1", caption: `ZeRO stage 1 splits only the optimizer state. Each GPU now owns one eighth of the purple block, and memory drops to ${g(at(c7, "zero1", 8))} GB. In the grid, each GPU's purple track is a single slice.`, apply: () => setup({ s: "zero1" }) },
    { id: "zero2", caption: `Stage 2 also keeps only this GPU's slice of the gradients: ${g(at(c7, "zero2", 8))} GB. Communication is still ${g(r7("zero2"), 2)} times DDP, because a reduce-scatter plus an all-gather moves the same bytes as one all-reduce.`, apply: () => setup({ s: "zero2" }) },
    { id: "zero3", caption: `Stage 3, which PyTorch calls FSDP, shards the weights too: ${g(at(c7, "zero3", 8))} GB per GPU. The price is gathering each layer's weights again in the backward pass, so traffic rises to ${g(r7("zero3"), 2)} times DDP.`, apply: () => setup({ s: "zero3" }) },
    { id: "scale-gpus", caption: `Sweep from 1 to 256 GPUs under ZeRO-1. The bar shrinks fast at first, then stalls near ${g(at(c7, "zero1", 256))} GB: replicated bf16 weights and gradients cost 4 bytes per parameter no matter how many GPUs you add.`, apply: () => setup({ s: "zero1", ni: 0 }), animate: (t) => { const v = Math.round(8 * t); if (v !== ni) setNi(v); }, animMs: 3200 },
    { id: "big-model", caption: `A 70B-style model needs ${g(at(c70, "ddp", 1) / 1000, 2)} TB of training state. Under ZeRO-3 it first fits in 80 GB per GPU at ${fit70} GPUs, where each GPU holds ${g(at(c70, "zero3", fit70 ?? 16))} GB.`, apply: () => setup({ model: "70b", s: "zero3", ni: GPU_COUNTS.indexOf(fit70 ?? 16) }) },
  ];

  const shown = Math.min(n, MAX_CELLS);
  const sh = sharded(strategy);
  return (
    <LabFrame
      id="zero-sharding"
      tour={tour}
      title="Data parallelism and ZeRO memory"
      subtitle="Synthetic Llama-like model shapes, 2,048 tokens per GPU with gradient checkpointing. Estimates of training state, not measurements."
      onReset={reset}
      presets={[
        { label: "7B on 8 GPUs", apply: reset },
        { label: "70B with FSDP", apply: () => setup({ model: "70b", s: "zero3", ni: 4 }) },
        { label: "13B on 24 GB cards", apply: () => setup({ model: "13b", s: "zero2", ni: 4, cap: "24" }) },
      ]}
      controls={
        <>
          <Segmented label="Model size" value={model} onChange={setModel} options={MODELS.map((k) => ({ value: k, label: PRESETS[k].label }))} />
          <Slider label="GPUs N" value={ni} min={0} max={GPU_COUNTS.length - 1} onChange={setNi} format={(i) => String(GPU_COUNTS[i])} />
          <Segmented label="Strategy" value={strategy} onChange={setStrategy} options={STRATEGIES.map((s) => ({ value: s.id, label: s.label }))} />
          <Segmented label="Precision" value={prec} onChange={setPrec} options={(Object.keys(BYTES) as Precision[]).map((k) => ({ value: k, label: BYTES[k].label }))} />
          <Segmented label="GPU memory" value={cap} onChange={setCap} options={[{ value: "24", label: "24 GB" }, { value: "48", label: "48 GB" }, { value: "80", label: "80 GB" }]} />
        </>
      }
      readout={
        <>
          <Stat label="parameters P" value={`${fmt(P / 1e9, 2)} B`} />
          <Stat label="bytes per param" value={`${BYTES[prec].w + BYTES[prec].g + BYTES[prec].o}`} />
          <Stat label={`${cur.label} per GPU`} value={`${fmt(cur.sum / GB, 2)} GB`} color={cur.sum / GB <= capGB ? "var(--c-green)" : "var(--c-red)"} />
          <Stat label="sent per GPU per step" value={`${fmt(cur.comm / GB, 2)} GB`} />
          <Stat label="comm vs DDP" value={`${fmt(ratio, 2)}×`} />
          <Stat label={`fewest GPUs to fit ${cap} GB`} value={fitN ?? "> 256"} />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">
            {cur.label} on {n} GPU{n === 1 ? "" : "s"}: <Tex>{`${BYTES[prec].w}P${sh.w ? "/N" : ""} + ${BYTES[prec].g}P${sh.g ? "/N" : ""} + ${BYTES[prec].o}P${sh.o ? "/N" : ""}`}</Tex> = {fmt(cur.sum / GB, 1)} GB per GPU with activations,
            {" "}{cur.sum / GB <= capGB ? `which fits in ${cap} GB.` : `which does not fit in ${cap} GB${fitN ? `; it needs ${fitN} GPUs` : ", even on 256 GPUs"}.`}
          </p>
          <p className="mt-1">
            {n === 1
              ? "With one GPU nothing can be sharded, so every strategy costs the same. Add GPUs to see the stages separate."
              : strategy === "ddp"
                ? "Plain data parallelism replicates everything: more GPUs add throughput but never reduce memory per GPU."
                : strategy === "zero3"
                  ? `Everything is sharded, so memory keeps falling as you add GPUs, but each step sends ${fmt(ratio, 2)}× the bytes of DDP because weights are gathered for the forward and again for the backward pass.`
                  : `The ${sh.g ? "gradients and optimizer states are" : "optimizer state is"} split ${n} ways at the same communication cost as DDP. The replicated ${sh.g ? "weights" : "weights and gradients"} set a floor that more GPUs cannot remove.`}
          </p>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Stacked memory per GPU for DDP and ZeRO stages" className="w-full h-auto select-none">
        <title>Memory per GPU by strategy</title>
        {ticks([0, xMax]).map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={M.t} y2={H - M.b} stroke="var(--grid)" />
            <text x={sx(t)} y={H - M.b + 16} textAnchor="middle" fontSize="11" fill="var(--faint)">{t}</text>
          </g>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" fontSize="12" fill="var(--muted)">memory per GPU (GB, estimate)</text>
        {rows.map((row, i) => {
          let left = 0;
          const y = M.t + i * rowH + rowH * 0.18, h = rowH * 0.64, on = row.id === strategy;
          return (
            <g key={row.id} opacity={on ? 1 : 0.4} onClick={() => setStrategy(row.id)} style={{ cursor: "pointer" }}>
              <text x={M.l - 8} y={y + h / 2 + 4} textAnchor="end" fontSize="12" fontWeight={on ? 600 : 400} fill="var(--text)">{row.label}</text>
              {PARTS.map((p) => {
                const v = row.mem[p.key] / GB, x0 = left;
                left += v;
                return <rect key={p.key} x={sx(x0)} y={y} width={Math.max(0, sx(x0 + v) - sx(x0))} height={h} fill={p.color} style={{ transition: "all 300ms" }}><title>{`${p.label}: ${fmt(v, 2)} GB`}</title></rect>;
              })}
              <text x={sx(left) + 6} y={y + h / 2 + 4} fontSize="11" fill="var(--text)">{fmt(left, 1)} GB</text>
            </g>
          );
        })}
        <line x1={sx(capGB)} x2={sx(capGB)} y1={M.t - 8} y2={H - M.b} stroke="var(--c-red)" strokeWidth={2} strokeDasharray="5 4" />
        <text x={sx(capGB)} y={M.t - 12} textAnchor="middle" fontSize="11" fill="var(--c-red)">{cap} GB GPU</text>
      </svg>
      <Legend items={PARTS.map((p) => ({ label: p.label, color: p.color }))} />

      <p className="mt-4 mb-1 text-xs text-muted">What each GPU stores under {cur.label}: a full bar is a replicated copy, a slice is that GPU&apos;s 1/{n} shard.</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2" role="img" aria-label={`Shards held by each of ${n} GPUs`}>
        {Array.from({ length: shown }, (_, r) => {
          const gpu = n > MAX_CELLS && r === shown - 1 ? n - 1 : r;
          return (
            <div key={r} className="rounded-md border border-line p-2">
              <div className="text-[0.7rem] text-muted mb-1">{n > MAX_CELLS && r === shown - 1 ? `… GPU ${gpu}` : `GPU ${gpu}`}</div>
              {PARTS.slice(0, 3).map((p) => {
                const isSh = sh[p.sh as "w" | "g" | "o"];
                return (
                  <svg key={p.key} viewBox="0 0 100 8" className="w-full h-2 mb-1 block" aria-hidden>
                    <rect x={0} y={0} width={100} height={8} rx={2} fill="var(--surface-2)" />
                    <rect x={isSh ? (100 * gpu) / n : 0} y={0} width={isSh ? Math.max(100 / n, 1.5) : 100} height={8} rx={2} fill={p.color} style={{ transition: "all 300ms" }} />
                  </svg>
                );
              })}
            </div>
          );
        })}
      </div>
      <div className="mt-3 grid gap-1 text-xs">
        {rows.map((row) => (
          <div key={row.id} className="flex items-center gap-2">
            <span className="w-24 text-muted">{row.label}</span>
            <svg viewBox="0 0 200 8" className="flex-1 h-2" aria-hidden>
              <rect x={0} y={0} width={(200 * (ddp.comm ? row.comm / ddp.comm : 1)) / 1.5} height={8} rx={2} fill={row.id === strategy ? "var(--accent)" : "var(--faint)"} />
            </svg>
            <span className="w-28 text-right font-mono">{n === 1 ? "no traffic" : `${fmt(ddp.comm ? row.comm / ddp.comm : 1, 2)}× DDP traffic`}</span>
          </div>
        ))}
      </div>
    </LabFrame>
  );
}
