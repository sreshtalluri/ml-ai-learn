"use client";
import { useEffect, useMemo, useState } from "react";
import { Button, LabFrame, Plot, Slider, Stat, Toggle } from "./ui";

// A fully MOCKED agent: the "model" is a script and the flight data is SYNTHETIC. Token counts are illustrative.
export const BASE_TOKENS = { system: 500, toolSchemas: 600, user: 100 }; // 1,200 tokens before step 1
export const BASE = BASE_TOKENS.system + BASE_TOKENS.toolSchemas + BASE_TOKENS.user;
const TASK = "Find the cheapest refundable SFO→SEA flight on Nov 14 under $300 and hold it.";

export type Opts = { toolError: boolean; ambiguous: boolean; maxSteps: number; approval: boolean };
export type Decision = "approve" | "reject" | undefined;
type Call = { name: string; args: Record<string, string | number> };
export type Turn = {
  kind: "ok" | "error" | "wrong" | "awaiting" | "declined" | "final" | "bad-final" | "stopped";
  thought: string;
  call?: Call;
  obs?: unknown;
  outTok: number; // tokens the model writes this turn (thought + call)
  obsTok: number; // tokens the observation adds to the context
};

/** Every model call re-reads the whole context: sum over t of (base + (t-1) * perStep). */
export function cumulativeInputTokens(base: number, perStep: number, steps: number) {
  return steps * base + (perStep * steps * (steps - 1)) / 2;
}

const SEARCH: Call = { name: "search_flights", args: { origin: "SFO", destination: "SEA", date: "2026-11-14", max_price_usd: 300 } };
const FLIGHTS = { flights: [{ flight_id: "UA412", price_usd: 189 }, { flight_id: "AS330", price_usd: 214 }, { flight_id: "DL1180", price_usd: 262 }] };
const hold = (id: string): Call => ({ name: "hold_booking", args: { flight_id: id, passenger_id: "traveler-1" } });

/** The scripted run for a given configuration. Pure, so the lesson's numbers are testable. */
export function buildTrace(o: Opts, decision?: Decision): Turn[] {
  const t: Turn[] = [];
  let target = "AS330";
  let price = 214;
  if (o.ambiguous) {
    t.push({ kind: "wrong", thought: "\"search\" sounds like the tool for finding flights.", call: { name: "search", args: { query: "cheapest refundable flight SFO to SEA Nov 14" } },
      obs: { results: [{ title: "Fare deals blog (March)", snippet: "DL1180 SFO-SEA from $199, fully refundable" }] }, outTok: 70, obsTok: 450 });
    target = "DL1180";
    price = 199;
  } else {
    if (o.toolError) {
      t.push({ kind: "error", thought: "I need candidate flights first.", call: SEARCH, obs: { error: "503 upstream timeout", retryable: true }, outTok: 80, obsTok: 40 });
      t.push({ kind: "ok", thought: "The error is marked retryable; retry the same call once (the runtime adds backoff).", call: SEARCH, obs: FLIGHTS, outTok: 90, obsTok: 320 });
    } else {
      t.push({ kind: "ok", thought: "I need candidate flights first.", call: SEARCH, obs: FLIGHTS, outTok: 80, obsTok: 320 });
    }
    t.push({ kind: "ok", thought: "UA412 is cheapest. Is it refundable?", call: { name: "get_fare_rules", args: { flight_id: "UA412" } }, obs: { refundable: false, change_fee_usd: 99 }, outTok: 60, obsTok: 60 });
    t.push({ kind: "ok", thought: "Not refundable. Check the next cheapest, AS330.", call: { name: "get_fare_rules", args: { flight_id: "AS330" } }, obs: { refundable: true, change_fee_usd: 0 }, outTok: 60, obsTok: 60 });
  }
  const holdThought = o.ambiguous ? `The blog says DL1180 is refundable at $${price}. Hold it.` : "AS330 is the cheapest refundable fare. Hold it.";
  const held = { hold_id: "H-7Q2", flight_id: target, price_usd: o.ambiguous ? 262 : 214, expires_in_hours: 24 };
  if (o.approval) {
    if (!decision) {
      t.push({ kind: "awaiting", thought: holdThought, call: hold(target), outTok: 70, obsTok: 0 });
    } else if (decision === "reject") {
      t.push({ kind: "declined", thought: holdThought, call: hold(target), obs: { status: "declined_by_user", executed: false }, outTok: 70, obsTok: 30 });
      t.push({ kind: "final", thought: "The user declined; report options without acting.", obs: undefined, outTok: 60, obsTok: 0 });
    } else {
      t.push({ kind: "ok", thought: holdThought, call: hold(target), obs: { approved_by: "user", ...held }, outTok: 70, obsTok: 90 });
    }
  } else {
    t.push({ kind: "ok", thought: holdThought, call: hold(target), obs: held, outTok: 70, obsTok: 90 });
  }
  if (!o.approval || decision === "approve") {
    t.push(o.ambiguous
      ? { kind: "bad-final", thought: `Done: held DL1180 at $${price}.`, outTok: 60, obsTok: 0 }
      : { kind: "final", thought: "Done: held AS330 at $214 (refundable, hold expires in 24 h).", outTok: 60, obsTok: 0 });
  }
  if (t.length > o.maxSteps) {
    return [...t.slice(0, o.maxSteps), { kind: "stopped", thought: `Step limit (${o.maxSteps}) reached. The runtime stops the loop and returns partial progress to the user.`, outTok: 0, obsTok: 0 }];
  }
  return t;
}

/** Input tokens each model call reads, given the turns before it. */
export function contextPerCall(turns: Turn[]) {
  const out: number[] = [];
  let ctx = BASE;
  for (const t of turns) {
    if (t.kind === "stopped") break;
    out.push(ctx);
    ctx += t.outTok + t.obsTok;
  }
  return out;
}

const KIND_COLOR: Record<Turn["kind"], string> = {
  ok: "var(--c-blue)", error: "var(--c-orange)", wrong: "var(--c-red)", awaiting: "var(--c-purple)", declined: "var(--c-purple)",
  final: "var(--c-green)", "bad-final": "var(--c-red)", stopped: "var(--c-orange)",
};
const DEFAULT: Opts = { toolError: false, ambiguous: false, maxSteps: 8, approval: false };
const json = (v: unknown) => JSON.stringify(v).replace(/,"/g, ', "').replace(/":/g, '": ');

export default function AgentLoopLab() {
  const [opts, setOpts] = useState<Opts>(DEFAULT);
  const [decision, setDecision] = useState<Decision>(undefined);
  const [shown, setShown] = useState(0);
  const [playing, setPlaying] = useState(false);

  const trace = useMemo(() => buildTrace(opts, decision), [opts, decision]);
  const visible = trace.slice(0, shown);
  const last = visible[visible.length - 1];
  const waiting = last?.kind === "awaiting";
  const done = shown >= trace.length && !waiting;

  const ctxAll = contextPerCall(trace);
  const ctxShown = ctxAll.slice(0, visible.filter((t) => t.kind !== "stopped").length);
  const ctxNow = visible.reduce((s, t) => s + t.outTok + t.obsTok, BASE);
  const billed = ctxShown.reduce((s, v) => s + v, 0);
  const calls = visible.filter((t) => t.call && t.kind !== "awaiting").length;
  const errors = visible.filter((t) => t.kind === "error" || t.kind === "wrong").length;

  const config = (patch: Partial<Opts>) => { setOpts((o) => ({ ...o, ...patch })); setDecision(undefined); setShown(0); setPlaying(false); };
  const preset = (o: Partial<Opts>) => () => { setOpts({ ...DEFAULT, ...o }); setDecision(undefined); setShown(0); setPlaying(true); };
  const step = () => setShown((s) => Math.min(s + 1, trace.length));
  const decide = (d: Decision) => setDecision(d); // the trace rebuilds and the pending turn resolves in place

  const running = playing && !waiting && shown < trace.length;
  useEffect(() => {
    if (!running) return;
    const id = setTimeout(() => setShown((s) => s + 1), 1100);
    return () => clearTimeout(id);
  }, [running, shown]);

  const maxCtx = Math.max(...ctxAll, BASE) * 1.1;
  const interpretation = (() => {
    if (!last) return "Press Step (or Autoplay). Before the first model call the context already holds the system prompt, every tool schema, and the user task: 1,200 tokens.";
    switch (last.kind) {
      case "error": return "The tool failed with a retryable error. Returning the error as an observation (instead of crashing) lets the model decide to retry; the runtime should cap retries and add backoff.";
      case "wrong": return "Ambiguous tool names and descriptions led the model to a general web search. It got a stale blog snippet instead of live inventory. The fix is in the tool layer: clear names, descriptions that say when not to use a tool.";
      case "awaiting": return "hold_booking has a side effect, so the runtime pauses and asks a human. The model proposed the call; nothing has executed yet.";
      case "declined": return "The user rejected the call. The runtime returns that as an observation, and the model must respect it rather than retrying.";
      case "stopped": return `The step budget ran out before the task finished. A max-steps limit bounds cost and runaway loops; the honest behavior is to stop and report partial progress.`;
      case "bad-final": return "The agent reports success, but the hold observation says DL1180 costs $262, and AS330 at $214 was cheaper and refundable. The final answer contradicts the agent's own observation; a verifier that checks claims against tool results would catch this.";
      case "final": return `Finished in ${ctxShown.length} model calls. Every call re-read the growing context, so ${billed.toLocaleString()} input tokens were billed for a context that ended at ${ctxNow.toLocaleString()}.`;
      default: return `Step ${visible.length}: the observation is appended to the context, which is now ${ctxNow.toLocaleString()} tokens. The next model call reads all of it.`;
    }
  })();

  return (
    <LabFrame
      id="agent-loop"
      title="Agent loop lab"
      subtitle={<>Task: “{TASK}” A scripted model with four mocked tools; step through each thought, tool call, and observation.</>}
      onReset={() => { setOpts(DEFAULT); setDecision(undefined); setShown(0); setPlaying(false); }}
      presets={[
        { label: "Happy path", apply: preset({}) },
        { label: "Flaky tool", apply: preset({ toolError: true }) },
        { label: "Ambiguous tools", apply: preset({ ambiguous: true }) },
        { label: "Step limit 3", apply: preset({ maxSteps: 3 }) },
        { label: "Human approval", apply: preset({ approval: true }) },
      ]}
      controls={
        <>
          <div className="flex flex-wrap gap-2">
            <Button primary onClick={step} disabled={done || waiting}>Step</Button>
            <Button onClick={() => setPlaying(!running)} disabled={done || waiting}>{running ? "Pause" : "Autoplay"}</Button>
          </div>
          {waiting && (
            <div className="rounded-lg border border-line p-2 space-y-2">
              <p className="text-[0.8rem] text-ink">Approve <span className="font-mono">hold_booking</span>?</p>
              <div className="flex gap-2">
                <Button primary onClick={() => decide("approve")}>Approve</Button>
                <Button onClick={() => decide("reject")}>Reject</Button>
              </div>
            </div>
          )}
          <Toggle label="Search tool returns an error" checked={opts.toolError} onChange={(v) => config({ toolError: v })} />
          <Toggle label="Ambiguous tool descriptions" checked={opts.ambiguous} onChange={(v) => config({ ambiguous: v })} />
          <Toggle label="Require approval for side effects" checked={opts.approval} onChange={(v) => config({ approval: v })} />
          <Slider label="max steps (model calls)" value={opts.maxSteps} min={2} max={8} onChange={(v) => config({ maxSteps: v })} />
        </>
      }
      readout={
        <>
          <Stat label="model calls" value={`${ctxShown.length} / ${opts.maxSteps}`} />
          <Stat label="tool calls run" value={calls} />
          <Stat label="errors / wrong tool" value={errors} color={errors ? "var(--c-orange)" : undefined} />
          <Stat label="context now" value={`${ctxNow.toLocaleString()} tok`} />
          <Stat label="input tokens billed" value={billed.toLocaleString()} />
        </>
      }
      interpretation={interpretation}
    >
      <div className="mb-3">
        <div className="flex justify-between text-xs text-muted"><span>Context window</span><span className="font-mono">{ctxNow.toLocaleString()} tokens</span></div>
        <div className="mt-1 flex h-3 overflow-hidden rounded bg-surface-2" role="img" aria-label={`context ${ctxNow} tokens`}>
          <div style={{ width: `${(BASE / maxCtx) * 100}%`, background: "var(--c-purple)" }} title="system + tool schemas + task" />
          {visible.map((t, i) => (
            <div key={i} style={{ width: `${((t.outTok + t.obsTok) / maxCtx) * 100}%`, background: KIND_COLOR[t.kind], opacity: i % 2 ? 0.65 : 0.9 }} />
          ))}
        </div>
        <p className="mt-1 text-xs text-faint">Purple: system prompt + tool schemas + task (1,200). Each later segment: one turn&apos;s thought, call, and observation.</p>
      </div>

      <ol className="space-y-2">
        <li className="rounded-lg border border-line px-3 py-2 text-sm">
          <span className="font-mono text-xs text-faint">tools</span>{" "}
          <span className="font-mono text-xs">{opts.ambiguous ? "search · find · get_fare_rules · hold_booking" : "search_flights · get_fare_rules · hold_booking · web_search"}</span>
          <p className="mt-1 text-xs text-muted">{opts.ambiguous
            ? "search: “Search for information.” find: “Find results.” (Which one has live fares?)"
            : "search_flights: “Live scheduled flights, sorted by price.” web_search: “General web pages. Not a source of live prices or fare rules.”"}</p>
        </li>
        {visible.map((t, i) => (
          <li key={i} className="rounded-lg border border-line px-3 py-2 text-sm" style={{ borderLeft: `4px solid ${KIND_COLOR[t.kind]}` }}>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="font-mono text-faint">{t.kind === "stopped" ? "runtime" : `step ${i + 1}`}</span>
              {t.kind !== "ok" && <span className="font-medium" style={{ color: KIND_COLOR[t.kind] }}>{t.kind.replace("-", " ")}</span>}
              {t.kind !== "stopped" && <span className="font-mono text-faint">read {ctxAll[i]?.toLocaleString()} tok</span>}
            </div>
            <p className="mt-1 italic text-muted">{t.thought}</p>
            {t.call && <pre className="mt-1 whitespace-pre-wrap break-all rounded bg-surface-2 px-2 py-1 font-mono text-xs text-ink">→ {json(t.call)}</pre>}
            {t.obs !== undefined && <pre className="mt-1 whitespace-pre-wrap break-all rounded px-2 py-1 font-mono text-xs text-muted border border-line">← {json(t.obs)}</pre>}
          </li>
        ))}
      </ol>

      {ctxShown.length > 0 && (
        <div className="mt-4">
          <Plot title="Input tokens read by each model call" width={560} height={180} x={[0, Math.max(ctxAll.length, 4) + 1]} y={[0, maxCtx]} xLabel="model call" yLabel="input tokens" margin={{ t: 10, r: 10, b: 36, l: 56 }}>
            {({ sx, sy }) => ctxShown.map((v, i) => (
              <rect key={i} x={sx(i + 1) - 12} width={24} y={sy(v)} height={sy(0) - sy(v)} fill="var(--c-blue)" opacity={0.8} />
            ))}
          </Plot>
        </div>
      )}
    </LabFrame>
  );
}
