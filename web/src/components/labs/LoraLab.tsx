"use client";
import { useState } from "react";
import { fmt } from "@/lib/ml";
import { LabFrame, Legend, linear, Segmented, Slider, Stat, Tex, ticks, Toggle, type TourStep } from "./ui";

// Pure arithmetic, mirrors guide/code/15-llms/fine_tuning_memory.py. Configs are SYNTHETIC, Llama-like shapes.
export interface ModelCfg { d: number; layers: number; ffMult: number; kvFrac: number; vocab: number }
export type Targets = "qv" | "attn" | "all";
export type Method = "full" | "lora" | "qlora";
export type Optim = "adamw" | "adam8";

export const PRESETS: Record<string, ModelCfg & { label: string }> = {
  "1b": { label: "1B-style", d: 2048, layers: 22, ffMult: 2.75, kvFrac: 1, vocab: 32000 },
  "7b": { label: "7B-style", d: 4096, layers: 32, ffMult: 2.6875, kvFrac: 1, vocab: 32000 },
  "13b": { label: "13B-style", d: 5120, layers: 40, ffMult: 2.7, kvFrac: 1, vocab: 32000 },
  "70b": { label: "70B-style", d: 8192, layers: 80, ffMult: 3.5, kvFrac: 0.125, vocab: 32000 },
};

const TARGET_SET: Record<Targets, string[]> = { qv: ["q", "v"], attn: ["q", "k", "v", "o"], all: ["q", "k", "v", "o", "gate", "up", "down"] };

export function matrices(c: ModelCfg): [string, number, number][] {
  const f = Math.round(c.d * c.ffMult), kv = Math.round(c.d * c.kvFrac);
  return [["q", c.d, c.d], ["k", c.d, kv], ["v", c.d, kv], ["o", c.d, c.d], ["gate", c.d, f], ["up", c.d, f], ["down", f, c.d]];
}

export function paramCounts(c: ModelCfg) {
  const block = matrices(c).reduce((s, [, i, o]) => s + i * o, 0) * c.layers;
  const embed = 2 * c.vocab * c.d; // input embedding + untied output head
  return { block, embed, total: block + embed };
}

export function loraParams(c: ModelCfg, r: number, targets: Targets) {
  const set = TARGET_SET[targets];
  return r * matrices(c).filter(([n]) => set.includes(n)).reduce((s, [, i, o]) => s + i + o, 0) * c.layers;
}

export function activationBytes(c: ModelCfg, tokens: number, checkpointing: boolean) {
  const perLayer = 34 * tokens * c.d; // rough rule of thumb, 16-bit, fused attention
  return checkpointing ? c.layers * 2 * tokens * c.d + perLayer : c.layers * perLayer;
}

const NF4 = 4.127 / 8; // bytes per weight incl. double-quantized block constants

export function memory(c: ModelCfg, method: Method, trainable: number, opts: { tokens: number; checkpointing: boolean; optim: Optim }) {
  const { block, embed, total } = paramCounts(c);
  const optBytes = opts.optim === "adamw" ? 12 : 6; // fp32 master 4 + m,v at 4+4 (or 1+1 in 8-bit)
  const n = method === "full" ? total : trainable;
  const base = method === "full" ? 0 : method === "lora" ? 2 * total : NF4 * block + 2 * embed;
  return { weights: base + 2 * n, grads: 2 * n, optimizer: optBytes * n, activations: activationBytes(c, opts.tokens, opts.checkpointing) };
}

const GB = 1e9;
const RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const PARTS = [
  { key: "weights", label: "weights", color: "var(--c-blue)" },
  { key: "grads", label: "gradients", color: "var(--c-orange)" },
  { key: "optimizer", label: "optimizer states", color: "var(--c-purple)" },
  { key: "activations", label: "activations", color: "var(--c-teal)" },
] as const;
const METHODS: { id: Method; label: string }[] = [{ id: "full", label: "Full fine-tune" }, { id: "lora", label: "LoRA (bf16 base)" }, { id: "qlora", label: "QLoRA (NF4 base)" }];
const W = 560, H = 230, M = { t: 24, r: 64, b: 36, l: 112 };

const big = (n: number) => (n >= 1e9 ? `${fmt(n / 1e9, 2)} B` : n >= 1e6 ? `${fmt(n / 1e6, 2)} M` : n.toLocaleString("en-US"));

export default function LoraLab() {
  const [preset, setPreset] = useState("7b");
  const [cfg, setCfg] = useState<ModelCfg>(PRESETS["7b"]);
  const [ri, setRi] = useState(4);
  const [targets, setTargets] = useState<Targets>("all");
  const [optim, setOptim] = useState<Optim>("adamw");
  const [tokens, setTokens] = useState(2048);
  const [ckpt, setCkpt] = useState(true);
  const [gpu, setGpu] = useState<"24" | "48" | "80">("24");

  const r = RANKS[ri];
  const { total } = paramCounts(cfg);
  const T = loraParams(cfg, r, targets);
  const opts = { tokens, checkpointing: ckpt, optim };
  const rows = METHODS.map((m) => {
    const mem = memory(cfg, m.id, T, opts);
    return { ...m, mem, sum: Object.values(mem).reduce((a, b) => a + b, 0) };
  });
  const gpuGB = Number(gpu);
  const fits = rows.filter((x) => x.sum / GB <= gpuGB).map((x) => x.label);
  const xMax = Math.max(gpuGB * 1.15, ...rows.map((x) => x.sum / GB)) * 1.05;
  const sx = linear([0, xMax], [M.l, W - M.r]);
  const rowH = (H - M.t - M.b) / 3;

  const pick = (k: string) => { setPreset(k); setCfg(PRESETS[k]); };
  const reset = () => { pick("7b"); setRi(4); setTargets("all"); setOptim("adamw"); setTokens(2048); setCkpt(true); setGpu("24"); };
  const full = rows[0], lora = rows[1], q = rows[2];
  const optShare = full.mem.optimizer / full.sum;

  // Guided tour (Watch mode + explainers). Each step sets the full state it needs, then animates.
  const setup = (o: { model?: string; ri?: number; tokens?: number; ckpt?: boolean; gpu?: "24" | "48" | "80" } = {}) => {
    pick(o.model ?? "7b"); setRi(o.ri ?? 4); setTargets("all"); setOptim("adamw"); setTokens(o.tokens ?? 2048); setCkpt(o.ckpt ?? true); setGpu(o.gpu ?? "24");
  };
  const tour: TourStep[] = [
    { id: "bars", caption: "Three ways to fine-tune a 7B-style model. Each bar stacks weights in blue, gradients in orange, optimizer states in purple and activations in teal; the dashed red line is a 24 GB GPU.", apply: () => setup() },
    { id: "full", caption: "Full fine-tuning trains all 6.7 billion weights. Every one needs a gradient and fp32 AdamW state, so purple alone is about 81 GB and the bar reaches 108.6 GB.", apply: () => setup() },
    { id: "rank", caption: "LoRA freezes the base and trains two thin matrices per layer. Sweep the rank from 1 to 256: trainable parameters grow 256×, yet the LoRA bar barely moves, because the frozen blue base dominates.", apply: () => setup({ ri: 0 }), animate: (t) => { const v = Math.round(8 * t); if (v !== ri) setRi(v); }, animMs: 3000 },
    { id: "qlora", caption: "QLoRA stores that frozen base in 4-bit NF4. The blue segment shrinks from 13.5 GB to 3.9 GB, and the whole job fits in about 5 GB.", apply: () => setup() },
    { id: "activations", caption: "Turn off gradient checkpointing and grow the micro-batch from 2,048 to 8,192 tokens. The teal activations swell to over 36 GB and push even QLoRA past the 24 GB line.", apply: () => setup({ ckpt: false }), animate: (t) => { const v = 2048 + Math.round((6144 * t) / 512) * 512; if (v !== tokens) setTokens(v); }, animMs: 2800 },
    { id: "big-model", caption: "Scale up to a 70B-style model on a 48 GB GPU. Full fine-tuning needs over a terabyte and LoRA about 145 GB, but QLoRA fits in about 43 GB.", apply: () => setup({ model: "70b", gpu: "48" }) },
  ];

  return (
    <LabFrame
      id="lora"
      tour={tour}
      title="LoRA fine-tuning memory"
      subtitle="Synthetic Llama-like model shapes. Estimates of training state, not measurements: real jobs add framework overhead and fragmentation."
      onReset={reset}
      presets={[
        { label: "7B on a 24 GB GPU", apply: reset },
        { label: "70B on 48 GB", apply: () => { pick("70b"); setGpu("48"); setRi(4); setTargets("all"); } },
        { label: "No checkpointing, 8K tokens", apply: () => { setCkpt(false); setTokens(8192); } },
      ]}
      controls={
        <>
          <Segmented label="Model size" value={preset} onChange={pick}
            options={[...Object.entries(PRESETS).map(([k, v]) => ({ value: k, label: v.label })), ...(preset === "custom" ? [{ value: "custom", label: "custom" }] : [])]} />
          <Slider label="hidden size d" value={cfg.d} min={1024} max={8192} step={256} onChange={(v) => { setCfg({ ...cfg, d: v }); setPreset("custom"); }} />
          <Slider label="layers L" value={cfg.layers} min={8} max={96} onChange={(v) => { setCfg({ ...cfg, layers: v }); setPreset("custom"); }} />
          <Slider label="LoRA rank r" value={ri} min={0} max={RANKS.length - 1} onChange={setRi} format={(i) => String(RANKS[i])} />
          <Segmented label="LoRA target matrices" value={targets} onChange={setTargets}
            options={[{ value: "qv", label: "q, v" }, { value: "attn", label: "q, k, v, o" }, { value: "all", label: "all linear" }]} />
          <Segmented label="Optimizer state precision" value={optim} onChange={setOptim}
            options={[{ value: "adamw", label: "fp32 AdamW" }, { value: "adam8", label: "8-bit Adam" }]} />
          <Slider label="tokens per micro-batch" value={tokens} min={512} max={16384} step={512} onChange={setTokens} format={(v) => v.toLocaleString("en-US")} />
          <Toggle label="Gradient checkpointing" checked={ckpt} onChange={setCkpt} />
          <Segmented label="GPU memory" value={gpu} onChange={setGpu} options={[{ value: "24", label: "24 GB" }, { value: "48", label: "48 GB" }, { value: "80", label: "80 GB" }]} />
        </>
      }
      readout={
        <>
          <Stat label="total params" value={big(total)} />
          <Stat label="LoRA trainable" value={big(T)} color="var(--c-orange)" />
          <Stat label="% of total" value={`${fmt((100 * T) / total, 3)}%`} />
          <Stat label="adapter file (bf16)" value={`${fmt((2 * T) / 1e6, 1)} MB`} />
          {rows.map((x) => <Stat key={x.id} label={x.label} value={`${fmt(x.sum / GB, 2)} GB`} color={x.sum / GB <= gpuGB ? "var(--c-green)" : "var(--c-red)"} />)}
        </>
      }
      interpretation={
        <>
          <p className="text-ink">
            <Tex>{"W = W_0 + \\tfrac{\\alpha}{r}BA"}</Tex>: rank {r} trains <Tex>{`r(d_{in}+d_{out})`}</Tex> parameters per matrix, {big(T)} in total.
            {" "}On a {gpu} GB GPU, {fits.length ? <>{fits.join(" and ")} fit{fits.length === 1 ? "s" : ""}.</> : "nothing fits: shard across GPUs or pick a smaller model."}
          </p>
          <p className="mt-1">
            In full fine-tuning, optimizer states are {fmt(100 * optShare, 0)}% of the bill. LoRA removes almost all gradients and optimizer state, so the frozen bf16 base ({fmt(lora.mem.weights / GB, 1)} GB) dominates; that is why changing the rank barely moves the LoRA bar.
            {" "}QLoRA stores that base in 4-bit NF4, cutting weights to {fmt(q.mem.weights / GB, 1)} GB.
            {!ckpt && " Without gradient checkpointing, activations grow with every layer and every token: watch the teal segment."}
          </p>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Stacked memory bars for full fine-tuning, LoRA and QLoRA" className="w-full h-auto select-none">
        <title>Training memory by method</title>
        {ticks([0, xMax]).map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={M.t} y2={H - M.b} stroke="var(--grid)" />
            <text x={sx(t)} y={H - M.b + 16} textAnchor="middle" fontSize="11" fill="var(--faint)">{t}</text>
          </g>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" fontSize="12" fill="var(--muted)">training memory (GB, estimate)</text>
        {rows.map((row, i) => {
          let left = 0;
          const y = M.t + i * rowH + rowH * 0.2, h = rowH * 0.6;
          return (
            <g key={row.id}>
              <text x={M.l - 8} y={y + h / 2 + 4} textAnchor="end" fontSize="12" fill="var(--text)">{row.label}</text>
              {PARTS.map((p) => {
                const v = row.mem[p.key] / GB, x0 = left;
                left += v;
                return <rect key={p.key} x={sx(x0)} y={y} width={Math.max(0, sx(x0 + v) - sx(x0))} height={h} fill={p.color} style={{ transition: "all 300ms" }}><title>{`${p.label}: ${fmt(v, 2)} GB`}</title></rect>;
              })}
              <text x={sx(left) + 6} y={y + h / 2 + 4} fontSize="11" fill="var(--text)">{fmt(left, 1)} GB</text>
            </g>
          );
        })}
        <line x1={sx(gpuGB)} x2={sx(gpuGB)} y1={M.t - 8} y2={H - M.b} stroke="var(--c-red)" strokeWidth={2} strokeDasharray="5 4" />
        <text x={sx(gpuGB)} y={M.t - 12} textAnchor="middle" fontSize="11" fill="var(--c-red)">{gpu} GB GPU</text>
      </svg>
      <Legend items={PARTS.map((p) => ({ label: p.label, color: p.color }))} />
    </LabFrame>
  );
}
