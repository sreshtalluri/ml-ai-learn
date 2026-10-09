"use client";
import { useState } from "react";
import { fmt } from "@/lib/ml";
import { LabFrame, Legend, Plot, Segmented, Slider, Stat } from "./ui";

// SYNTHETIC model shapes and accelerator specs, in the range of current hardware but not any specific product.
// GB means 10^9 bytes. Activations, framework overhead, and fragmentation are not included.
export interface ModelShape { id: string; label: string; layers: number; heads: number; kvHeads: number; dHead: number; params: number }
export const MODELS: ModelShape[] = [
  { id: "small", label: "Small MHA (1.3B-style)", layers: 24, heads: 32, kvHeads: 32, dHead: 64, params: 1.3e9 },
  { id: "8b", label: "8B-style GQA", layers: 32, heads: 32, kvHeads: 8, dHead: 128, params: 8e9 },
  { id: "70b", label: "70B-style GQA", layers: 80, heads: 64, kvHeads: 8, dHead: 128, params: 70e9 },
];
export const GPUS = [
  { id: "24", label: "24 GB · 1.0 TB/s", memGB: 24, bwTBs: 1.0 },
  { id: "80", label: "80 GB · 3.0 TB/s", memGB: 80, bwTBs: 3.0 },
  { id: "144", label: "144 GB · 5.0 TB/s", memGB: 144, bwTBs: 5.0 },
];
type Attn = "mha" | "gqa" | "mqa";
type Bits = "16" | "8" | "4";

/** Bytes of K and V stored per token per sequence: 2 · L · n_kv · d_head · bytes. */
export const kvBytesPerToken = (layers: number, nKv: number, dHead: number, bytes: number) => 2 * layers * nKv * dHead * bytes;

export function kvHeadsFor(m: ModelShape, attn: Attn) {
  return attn === "mha" ? m.heads : attn === "mqa" ? 1 : Math.min(8, m.heads);
}

export function inferenceBudget(o: {
  model: ModelShape; nKv: number; ctx: number; batch: number; weightBytes: number; kvBytes: number; memGB: number; bwTBs: number; gpus: number;
}) {
  const perTok = kvBytesPerToken(o.model.layers, o.nKv, o.model.dHead, o.kvBytes);
  const weights = o.model.params * o.weightBytes;
  const kvPerSeq = perTok * o.ctx;
  const kv = kvPerSeq * o.batch;
  const capacity = o.memGB * 1e9 * o.gpus;
  const bw = o.bwTBs * 1e12 * o.gpus; // ideal scaling across GPUs; ignores communication
  // Upper bound: each decode step reads all weights once plus every sequence's KV cache.
  const stepsPerS = bw / (weights + kv);
  return {
    perTokBytes: perTok,
    weightsGB: weights / 1e9,
    kvGB: kv / 1e9,
    totalGB: (weights + kv) / 1e9,
    capacityGB: capacity / 1e9,
    fits: weights + kv <= capacity,
    maxBatch: Math.max(0, Math.floor((capacity - weights) / kvPerSeq)),
    perSeqTps: stepsPerS,
    aggTps: stepsPerS * o.batch,
  };
}

const BYTES: Record<Bits, number> = { "16": 2, "8": 1, "4": 0.5 };
const DEFAULT = { modelId: "8b", attn: "gqa" as Attn, exp: 13, batch: 16, wBits: "16" as Bits, kvBits: "16" as Bits, gpuId: "80", gpus: 1 };

export default function KvCacheLab() {
  const [s, setS] = useState(DEFAULT);
  const set = (p: Partial<typeof DEFAULT>) => setS((x) => ({ ...x, ...p }));
  const model = MODELS.find((m) => m.id === s.modelId)!;
  const gpu = GPUS.find((g) => g.id === s.gpuId)!;
  const nKv = kvHeadsFor(model, s.attn);
  const ctx = 2 ** s.exp;
  const args = { model, nKv, ctx, batch: s.batch, weightBytes: BYTES[s.wBits], kvBytes: BYTES[s.kvBits], memGB: gpu.memGB, bwTBs: gpu.bwTBs, gpus: s.gpus };
  const r = inferenceBudget(args);
  const mha = inferenceBudget({ ...args, nKv: model.heads });

  const xMax = 128;
  const yMax = Math.max(r.capacityGB, r.totalGB) * 1.15;
  const slope = (r.perTokBytes * 1024 * s.batch) / 1e9; // GB per thousand tokens of context
  const xEnd = Math.min(xMax, (yMax - r.weightsGB) / slope); // clip the line at the top of the plot
  const kvShare = r.kvGB / r.totalGB;

  return (
    <LabFrame
      id="kv-cache"
      title="KV cache and inference memory"
      subtitle="Synthetic model shapes and accelerator specs (illustrative, not specific products). GB = 10⁹ bytes; activations and overhead not included."
      onReset={() => setS(DEFAULT)}
      presets={[
        { label: "Lesson example (8B, 8k × 16)", apply: () => setS(DEFAULT) },
        { label: "70B on one GPU", apply: () => setS({ ...DEFAULT, modelId: "70b", exp: 13, batch: 8 }) },
        { label: "Small MHA, long context", apply: () => setS({ ...DEFAULT, modelId: "small", attn: "mha", exp: 15, batch: 16, gpuId: "24" }) },
        { label: "INT4 weights + FP8 KV", apply: () => setS({ ...DEFAULT, wBits: "4", kvBits: "8", gpuId: "24", batch: 8 }) },
      ]}
      controls={
        <>
          <label className="block text-[0.8rem] text-muted">Model
            <select value={s.modelId} onChange={(e) => { const m = MODELS.find((x) => x.id === e.target.value)!; set({ modelId: m.id, attn: m.kvHeads === m.heads ? "mha" : "gqa" }); }} className="mt-1 w-full rounded-lg border border-line bg-bg px-2 py-1.5 text-sm text-ink">
              {MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          </label>
          <Segmented label={`Attention (n_kv of ${model.heads} heads)`} value={s.attn} onChange={(v) => set({ attn: v })}
            options={[{ value: "mha", label: `MHA (${model.heads})` }, { value: "gqa", label: `GQA (${Math.min(8, model.heads)})` }, { value: "mqa", label: "MQA (1)" }]} />
          <Slider label="context length T" value={s.exp} min={9} max={17} onChange={(v) => set({ exp: v })} format={(v) => `${(2 ** v).toLocaleString()} tokens`} />
          <Slider label="batch size B" value={s.batch} min={1} max={64} onChange={(v) => set({ batch: v })} />
          <Segmented label="Weight precision" value={s.wBits} onChange={(v) => set({ wBits: v })} options={[{ value: "16", label: "BF16" }, { value: "8", label: "FP8/INT8" }, { value: "4", label: "INT4" }]} />
          <Segmented label="KV-cache precision" value={s.kvBits} onChange={(v) => set({ kvBits: v })} options={[{ value: "16", label: "BF16" }, { value: "8", label: "FP8" }, { value: "4", label: "4-bit" }]} />
          <Segmented label="Accelerator" value={s.gpuId} onChange={(v) => set({ gpuId: v })} options={GPUS.map((g) => ({ value: g.id, label: g.label }))} />
          <Segmented label="Number of GPUs (ideal tensor parallel)" value={String(s.gpus)} onChange={(v) => set({ gpus: Number(v) })} options={["1", "2", "4", "8"].map((v) => ({ value: v, label: v }))} />
        </>
      }
      readout={
        <>
          <Stat label="KV per token" value={`${(r.perTokBytes / 1024).toLocaleString()} KiB`} />
          <Stat label="weights" value={`${fmt(r.weightsGB, 2)} GB`} color="var(--c-purple)" />
          <Stat label="KV cache" value={`${fmt(r.kvGB, 2)} GB`} color="var(--c-blue)" />
          <Stat label="total / capacity" value={`${fmt(r.totalGB, 1)} / ${fmt(r.capacityGB, 0)} GB`} />
          <Stat label="fits?" value={r.fits ? "yes" : "no"} color={r.fits ? "var(--c-green)" : "var(--c-red)"} />
          <Stat label="max batch at this T" value={r.maxBatch} />
          <Stat label="decode ≤ per sequence" value={`${fmt(r.perSeqTps, 1)} tok/s`} color="var(--c-orange)" />
          <Stat label="decode ≤ aggregate" value={`${Math.round(r.aggTps).toLocaleString()} tok/s`} color="var(--c-teal)" />
        </>
      }
      interpretation={
        <>
          <p className="text-ink">
            KV cache = 2 · {model.layers} layers · {nKv} KV heads · {model.dHead} dims · {BYTES[s.kvBits]} bytes · {ctx.toLocaleString()} tokens · {s.batch} sequences = {fmt(r.kvGB, 2)} GB.{" "}
            {r.fits
              ? `It fits with ${fmt(r.capacityGB - r.totalGB, 1)} GB to spare (minus activations and overhead), and the KV cache is ${Math.round(kvShare * 100)}% of the memory used.`
              : `It does not fit: ${fmt(r.totalGB - r.capacityGB, 1)} GB over. ${r.weightsGB > r.capacityGB ? "Even the weights alone are too big; quantize them or shard across more GPUs." : `At this context only ${r.maxBatch} sequence${r.maxBatch === 1 ? "" : "s"} fit; lower the batch, quantize the KV cache, or add GPUs.`}`}
          </p>
          <p className="mt-1">
            Each decode step reads every weight and every cached key and value once, so speed is capped at bandwidth ÷ bytes read: about {fmt(r.perSeqTps, 0)} tokens/s for each user and {Math.round(r.aggTps).toLocaleString()} tokens/s in total.{" "}
            {s.attn !== "mha" && model.heads !== nKv ? `With full multi-head attention the same KV cache would be ${fmt(mha.kvGB, 1)} GB (${model.heads / nKv}× larger).` : model.kvHeads !== model.heads ? "" : "This model uses one KV head per query head, so its cache per token is large for its size."}
          </p>
        </>
      }
    >
      <Plot x={[0, xMax]} y={[0, yMax]} xLabel="context length T (thousand tokens)" yLabel="memory (GB)" title="Weights plus KV cache versus context length, with accelerator capacity">
        {({ sx, sy }) => (
          <>
            <rect x={sx(0)} y={sy(r.weightsGB)} width={sx(xMax) - sx(0)} height={sy(0) - sy(r.weightsGB)} fill="var(--c-purple)" opacity={0.12} />
            <line x1={sx(0)} x2={sx(xMax)} y1={sy(r.weightsGB)} y2={sy(r.weightsGB)} stroke="var(--c-purple)" strokeWidth={2} />
            <line x1={sx(0)} x2={sx(xEnd)} y1={sy(r.weightsGB)} y2={sy(r.weightsGB + slope * xEnd)} stroke="var(--c-blue)" strokeWidth={2.5} />
            <line x1={sx(0)} x2={sx(xMax)} y1={sy(r.capacityGB)} y2={sy(r.capacityGB)} stroke="var(--c-red)" strokeWidth={2} strokeDasharray="6 4" />
            <line x1={sx(ctx / 1024)} x2={sx(ctx / 1024)} y1={sy(0)} y2={sy(r.totalGB)} stroke="var(--c-orange)" strokeDasharray="3 3" />
            <circle cx={sx(ctx / 1024)} cy={sy(r.totalGB)} r={6} fill={r.fits ? "var(--c-green)" : "var(--c-red)"} stroke="var(--c-orange)" strokeWidth={2} />
          </>
        )}
      </Plot>
      <Legend items={[
        { label: "weights", color: "var(--c-purple)" },
        { label: `weights + KV cache, batch ${s.batch}`, color: "var(--c-blue)" },
        { label: `capacity (${fmt(r.capacityGB, 0)} GB)`, color: "var(--c-red)", dashed: true },
        { label: "current context", color: "var(--c-orange)", dashed: true },
      ]} />
    </LabFrame>
  );
}
