"use client";
import { useEffect, useMemo, useState } from "react";
import { fmt } from "@/lib/ml";
import { Button, LabFrame, Legend, Plot, Segmented, Slider, Stat, type TourStep } from "./ui";

// A SYNTHETIC 5x5 cliff grid, identical to guide/code/26-reinforcement-learning/q_learning.py.
// S = start, G = goal, P = pit (-100, episode ends), # = wall. Every other move costs -1.
export const LAYOUT = [
  ".....",
  ".#.#.",
  ".....",
  ".....",
  "SPPPG",
];
export const ROWS = 5, COLS = 5;
export const ACTIONS = ["up", "right", "down", "left"] as const;
const MOVES: [number, number][] = [[-1, 0], [0, 1], [1, 0], [0, -1]];
export const START = 4 * COLS + 0, GOAL = 4 * COLS + 4;
export const STEP_REWARD = -1, PIT_REWARD = -100, MAX_STEPS = 100;
const cell = (s: number) => LAYOUT[Math.floor(s / COLS)][s % COLS];

export type Algo = "q" | "sarsa";
export type Params = { algo: Algo; alpha: number; gamma: number; eps: number };
export type Update = { s: number; a: number; r: number; s2: number; done: boolean; boot: number; target: number; old: number; next: number; alpha: number; gamma: number };
export type Sim = {
  Q: number[];            // ROWS*COLS*4 action values, index s*4 + a
  s: number;              // agent's cell
  a: number | null;       // SARSA: the next action, already chosen
  seed: number;           // PRNG state (mulberry32), so every run is reproducible
  epReturn: number; epSteps: number;
  returns: number[];      // return of each finished episode
  last: Update | null;
};

/** Deterministic move. Walls and edges leave you in place. */
export function envStep(s: number, a: number): { s2: number; r: number; done: boolean } {
  const r = Math.floor(s / COLS), c = s % COLS;
  let nr = r + MOVES[a][0], nc = c + MOVES[a][1];
  if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || LAYOUT[nr][nc] === "#") { nr = r; nc = c; }
  const s2 = nr * COLS + nc, k = LAYOUT[nr][nc];
  if (k === "P") return { s2, r: PIT_REWARD, done: true };
  return { s2, r: STEP_REWARD, done: k === "G" };
}

/** The TD target. Q-learning bootstraps from the best next action; SARSA from the action it will actually take. */
export function tdTarget(algo: Algo, r: number, gamma: number, qNext: number[], aNext: number, done: boolean) {
  if (done) return r;
  return r + gamma * (algo === "q" ? Math.max(...qNext) : qNext[aNext]);
}

function rand(sim: Sim) { // mulberry32, state kept on the sim
  sim.seed = (sim.seed + 0x6d2b79f5) >>> 0;
  let t = sim.seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const qs = (Q: number[], s: number) => Q.slice(s * 4, s * 4 + 4);

export function epsGreedy(sim: Sim, s: number, eps: number) {
  if (rand(sim) < eps) return Math.floor(rand(sim) * 4);
  const q = qs(sim.Q, s), best = Math.max(...q);
  const ties = [0, 1, 2, 3].filter((a) => q[a] === best);
  return ties[Math.floor(rand(sim) * ties.length)];
}

export const newSim = (seed = 7): Sim => ({ Q: Array(ROWS * COLS * 4).fill(0), s: START, a: null, seed, epReturn: 0, epSteps: 0, returns: [], last: null });

/** One environment step plus one TD update. Mutates sim (use advance() for a copy). */
function stepMut(sim: Sim, p: Params) {
  const s = sim.s;
  const a = sim.a ?? epsGreedy(sim, s, p.eps);
  const { s2, r, done } = envStep(s, a);
  const a2 = done ? 0 : epsGreedy(sim, s2, p.eps); // SARSA uses it; Q-learning re-picks next step (same rule)
  const qNext = qs(sim.Q, s2);
  const target = tdTarget(p.algo, r, p.gamma, qNext, a2, done);
  const old = sim.Q[s * 4 + a];
  sim.Q[s * 4 + a] = old + p.alpha * (target - old);
  sim.last = { s, a, r, s2, done, boot: done ? 0 : p.algo === "q" ? Math.max(...qNext) : qNext[a2], target, old, next: sim.Q[s * 4 + a], alpha: p.alpha, gamma: p.gamma };
  sim.epReturn += r;
  sim.epSteps += 1;
  if (done || sim.epSteps >= MAX_STEPS) { // a time limit is not a terminal state: we just start over
    sim.returns.push(sim.epReturn);
    sim.s = START; sim.a = null; sim.epReturn = 0; sim.epSteps = 0;
    return true;
  }
  sim.s = s2;
  sim.a = p.algo === "sarsa" ? a2 : null;
  return false;
}

const clone = (sim: Sim): Sim => ({ ...sim, Q: sim.Q.slice(), returns: sim.returns.slice() });

/** Take n single steps on a copy. */
export function advance(sim: Sim, p: Params, n = 1) {
  const out = clone(sim);
  for (let i = 0; i < n; i++) stepMut(out, p);
  return out;
}

/** Run until n more episodes have finished, on a copy. */
export function runEpisodes(sim: Sim, p: Params, n: number) {
  const out = clone(sim);
  const target = out.returns.length + n;
  while (out.returns.length < target) stepMut(out, p);
  return out;
}

export const train = (p: Params, episodes: number, seed = 7) => runEpisodes(newSim(seed), p, episodes);

/** Cells visited by following the greedy policy from the start (first best action, no ties broken at random). */
export function greedyPath(Q: number[], maxLen = 24) {
  const path = [START];
  let s = START;
  while (cell(s) !== "P" && cell(s) !== "G" && path.length <= maxLen) {
    const q = qs(Q, s);
    s = envStep(s, q.indexOf(Math.max(...q))).s2;
    path.push(s);
  }
  return path;
}
/** Moves on the greedy path, or null if it loops or falls in a pit. */
export function pathMoves(Q: number[]) {
  const p = greedyPath(Q);
  return cell(p[p.length - 1]) === "G" ? p.length - 1 : null;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const DEFAULT: Params = { algo: "q", alpha: 0.2, gamma: 1, eps: 0.1 };
const ALGO_NAME: Record<Algo, string> = { q: "Q-learning", sarsa: "SARSA" };
const ARROW = ["↑", "→", "↓", "←"];
const rc = (s: number) => `(${Math.floor(s / COLS)},${s % COLS})`;

export default function QLearningLab() {
  const [p, setP] = useState<Params>(DEFAULT);
  const [sim, setSim] = useState<Sim>(() => newSim());
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = setTimeout(() => setSim((s) => advance(s, p)), 90);
    return () => clearTimeout(id);
  }, [playing, sim, p]);

  const set = (patch: Partial<Params>) => setP((o) => ({ ...o, ...patch }));
  const restart = (np: Params, episodes = 0) => { setP(np); setSim(episodes ? train(np, episodes) : newSim()); setPlaying(false); };
  const switchAlgo = (algo: Algo) => restart({ ...p, algo }); // different update rule, so start a fresh table

  const moves = pathMoves(sim.Q);
  const path = greedyPath(sim.Q);
  const recent = sim.returns.slice(-20);
  const n = sim.returns.length;

  // Guided tour. Numbers come from real runs, so the captions always match the screen.
  const ref = useMemo(() => {
    const q = train({ ...DEFAULT, algo: "q" }, 300), s = train({ ...DEFAULT, algo: "sarsa" }, 300);
    const first = train(DEFAULT, 1);
    const explore = train({ ...DEFAULT, algo: "sarsa", eps: 0.3 }, 300), qExplore = train({ ...DEFAULT, eps: 0.3 }, 300);
    return { q, s, first, explore, qExplore };
  }, []);
  const lastMean = (x: Sim) => fmt(mean(x.returns.slice(-50)), 1);
  const grow = (np: Params, total: number) => (t: number) => setSim(train(np, Math.max(1, Math.round(t * total))));
  const tour: TourStep[] = [
    { id: "world", caption: "Start bottom left, goal bottom right, three pits in between. Every move costs 1 point and a pit costs 100 and ends the episode. All Q-values start at zero, so the agent knows nothing.", apply: () => restart(DEFAULT) },
    { id: "first-episode", caption: `The first episode ${ref.first.returns[0] <= PIT_REWARD ? "ends in a pit" : "stumbles onto the goal"} with return ${ref.first.returns[0]}. Each move nudges one Q-value toward reward plus the discounted best value next door, so only visited cells change.`, apply: () => restart(DEFAULT, 1) },
    { id: "learning", caption: `Run 300 episodes. The curve climbs as values spread back from the goal; arrows grow toward the best move in every cell. The last 50 episodes average ${lastMean(ref.q)} while still exploring 10% of the time.`, apply: () => restart(DEFAULT, 1), animate: grow(DEFAULT, 300), animMs: 3500 },
    { id: "q-path", caption: `Q-learning's greedy path takes ${pathMoves(ref.q.Q) ?? "no"} moves, hugging the pits. It learns the value of acting greedily later, even though during training a random step next to the pits sometimes costs 100.`, apply: () => restart(DEFAULT, 300) },
    { id: "sarsa", caption: `SARSA learns the value of what it actually does, exploration included. Its greedy path takes ${pathMoves(ref.s.Q) ?? "no"} moves, a row away from the pits, and it averages ${lastMean(ref.s)} while exploring versus ${lastMean(ref.q)} for Q-learning.`, apply: () => restart({ ...DEFAULT, algo: "sarsa" }, 1), animate: grow({ ...DEFAULT, algo: "sarsa" }, 300), animMs: 3000 },
    { id: "explore-more", caption: `Now raise exploration to 30% for both. Q-learning still picks the ${pathMoves(ref.qExplore.Q) ?? "no"}-move edge path, but while exploring it averages ${lastMean(ref.qExplore)}, against ${lastMean(ref.explore)} for SARSA. Q-learning learns the best plan; SARSA learns the best plan for an agent that keeps exploring.`, apply: () => restart({ ...DEFAULT, eps: 0.3 }, 300) },
  ];

  const last = sim.last;
  const interpretation = (() => {
    if (!n && !sim.epSteps) return "All Q-values start at 0. Press Step to watch one move and one update, or Run 50 episodes to see the values spread back from the goal.";
    if (moves === null) return `After ${n} episode${n === 1 ? "" : "s"} the greedy policy does not reach the goal yet: values have only spread to cells the agent has visited. Keep training.`;
    if (p.eps === 0) return `With ε = 0 the agent never explores on purpose. It still found a ${moves}-move path because unvisited moves keep their optimistic 0, so trying them looks attractive until each one is tried.`;
    if (p.algo === "q") return `Q-learning's greedy path takes ${moves} moves${moves === 6 ? ", the shortest route along the pits" : ""}. Its target uses the max over next actions, so it learns the value of the greedy policy even while the agent explores, and pays for it with occasional falls (recent mean return ${fmt(mean(recent), 1)}).`;
    return `SARSA's greedy path takes ${moves} moves${moves > 6 ? `, ${moves - 6} more than the shortest, keeping clear of the pits` : ""}. Its target uses the action it will actually take, exploration included, so cells next to the pits look worse. Recent mean return while exploring: ${fmt(mean(recent), 1)}.`;
  })();

  // ----- drawing -----
  const S = 64, PAD = 4, W = COLS * S + 2 * PAD, H = ROWS * S + 2 * PAD;
  const cx = (s: number) => PAD + (s % COLS) * S + S / 2, cy = (s: number) => PAD + Math.floor(s / COLS) * S + S / 2;
  const xMax = Math.max(50, Math.ceil(n / 50) * 50);
  const smooth = sim.returns.map((_, i) => mean(sim.returns.slice(Math.max(0, i - 19), i + 1)));
  const clampY = (v: number) => Math.max(-120, v);

  return (
    <LabFrame
      id="q-learning"
      title="Q-learning on a cliff grid"
      subtitle="A synthetic 5×5 world: −1 per move, −100 for a pit. Watch the Q-table fill in, one temporal-difference update at a time."
      tour={tour}
      onReset={() => restart(DEFAULT)}
      presets={[
        { label: "Q-learning, 300 episodes", apply: () => restart({ ...DEFAULT, algo: "q" }, 300) },
        { label: "SARSA, 300 episodes", apply: () => restart({ ...DEFAULT, algo: "sarsa" }, 300) },
        { label: "No exploration (ε = 0)", apply: () => restart({ ...DEFAULT, eps: 0 }, 100) },
        { label: "Big steps (α = 1)", apply: () => restart({ ...DEFAULT, alpha: 1 }, 100) },
      ]}
      controls={
        <>
          <Segmented label="update rule" value={p.algo} onChange={switchAlgo}
            options={[{ value: "q", label: "Q-learning (max)" }, { value: "sarsa", label: "SARSA (next action)" }]} />
          <div className="flex flex-wrap gap-2">
            <Button primary onClick={() => setSim((s) => advance(s, p))}>Step</Button>
            <Button onClick={() => setSim((s) => runEpisodes(s, p, 1))}>Episode</Button>
            <Button onClick={() => setSim((s) => runEpisodes(s, p, 50))}>Run 50 episodes</Button>
            <Button onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Autoplay"}</Button>
          </div>
          <Slider label="learning rate α" value={p.alpha} min={0.05} max={1} step={0.05} onChange={(v) => set({ alpha: v })} format={(v) => v.toFixed(2)} hint="how far each update moves toward the target" />
          <Slider label="discount γ" value={p.gamma} min={0.5} max={1} step={0.01} onChange={(v) => set({ gamma: v })} format={(v) => v.toFixed(2)} />
          <Slider label="exploration ε" value={p.eps} min={0} max={0.5} step={0.01} onChange={(v) => set({ eps: v })} format={(v) => v.toFixed(2)} hint="chance of a random move" />
        </>
      }
      readout={
        <>
          <Stat label="episodes" value={n} />
          <Stat label="steps this episode" value={sim.epSteps} />
          <Stat label="last return" value={n ? sim.returns[n - 1] : "–"} />
          <Stat label="mean return (last 20)" value={recent.length ? fmt(mean(recent), 1) : "–"} color="var(--c-blue)" />
          <Stat label="greedy path" value={moves === null ? "doesn't reach goal" : `${moves} moves`} color="var(--c-teal)" />
          <Stat label="max Q at start" value={fmt(Math.max(...qs(sim.Q, START)), 2)} />
        </>
      }
      interpretation={interpretation}
    >
      <div className="grid gap-4 @2xl:grid-cols-2">
        <div>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Grid world with Q-value arrows in every cell" className="w-full h-auto max-w-[360px] mx-auto">
            <title>Q-values per cell: longer, darker arrows are better actions</title>
            {LAYOUT.flatMap((row, r) => [...row].map((k, c) => {
              const s = r * COLS + c, q = qs(sim.Q, s), best = Math.max(...q), worst = Math.min(...q);
              const touched = q.some((v) => v !== 0);
              const shade = touched ? Math.round(6 + 34 * Math.max(0, 1 + best / 20)) : 0;
              const fill = k === "#" ? "var(--muted)" : k === "P" ? "color-mix(in srgb, var(--c-red) 28%, var(--surface))"
                : k === "G" ? "color-mix(in srgb, var(--c-green) 30%, var(--surface))" : `color-mix(in srgb, var(--c-blue) ${shade}%, var(--surface))`;
              return (
                <g key={s}>
                  <rect x={PAD + c * S} y={PAD + r * S} width={S} height={S} fill={fill} stroke="var(--line, var(--border))" />
                  {k === "P" && <text x={cx(s)} y={cy(s) + 4} textAnchor="middle" fontSize="11" fill="var(--c-red)">pit −100</text>}
                  {k === "G" && <text x={cx(s)} y={cy(s) + 4} textAnchor="middle" fontSize="11" fill="var(--text)">goal</text>}
                  {(k === "." || k === "S") && touched && [0, 1, 2, 3].map((a) => {
                    const rel = best === worst ? 1 : (q[a] - worst) / (best - worst);
                    const len = S * (0.1 + 0.26 * rel), [dy, dx] = MOVES[a];
                    const x1 = cx(s) + dx * 6, y1 = cy(s) + dy * 6, x2 = cx(s) + dx * (6 + len), y2 = cy(s) + dy * (6 + len);
                    const isBest = q[a] === best;
                    return <line key={a} x1={x1} y1={y1} x2={x2} y2={y2} stroke={isBest ? "var(--c-teal)" : "var(--faint)"} strokeWidth={isBest ? 3 : 1.5} strokeLinecap="round" markerEnd={isBest ? "url(#ql-head)" : undefined} />;
                  })}
                  {(k === "." || k === "S") && <text x={PAD + c * S + 3} y={PAD + r * S + 11} fontSize="9" fill="var(--faint)">{k === "S" ? "S " : ""}{touched ? fmt(best, 1) : ""}</text>}
                </g>
              );
            }))}
            <defs>
              <marker id="ql-head" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="4" markerHeight="4" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--c-teal)" />
              </marker>
            </defs>
            {moves !== null && <polyline points={path.map((s) => `${cx(s)},${cy(s)}`).join(" ")} fill="none" stroke="var(--c-orange)" strokeWidth={2} strokeDasharray="5 4" opacity={0.85} />}
            <circle cx={cx(sim.s)} cy={cy(sim.s)} r={9} fill="var(--c-purple)" stroke="var(--surface)" strokeWidth={2} />
          </svg>
          <Legend items={[{ label: "best action (greedy policy)", color: "var(--c-teal)" }, { label: "greedy path", color: "var(--c-orange)", dashed: true }, { label: "agent", color: "var(--c-purple)" }]} />
        </div>
        <div className="min-w-0">
          <Plot title="Return per episode" x={[0, xMax]} y={[-120, 0]} xLabel="episode" yLabel="return" width={420} height={260} margin={{ t: 10, r: 10, b: 40, l: 48 }}>
            {({ sx, sy }) => (
              <>
                <line x1={sx(0)} x2={sx(xMax)} y1={sy(-6)} y2={sy(-6)} stroke="var(--c-green)" strokeDasharray="4 3" />
                <text x={sx(xMax) - 4} y={sy(-6) - 4} textAnchor="end" fontSize="10" fill="var(--c-green)">optimal −6</text>
                {n <= 600 && sim.returns.map((v, i) => <circle key={i} cx={sx(i + 1)} cy={sy(clampY(v))} r={1.6} fill="var(--faint)" />)}
                {n > 1 && <polyline fill="none" stroke="var(--c-blue)" strokeWidth={2} points={smooth.map((v, i) => `${sx(i + 1)},${sy(clampY(v))}`).join(" ")} />}
              </>
            )}
          </Plot>
          <Legend items={[{ label: "20-episode mean", color: "var(--c-blue)" }, { label: "one episode (clipped at −120)", color: "var(--faint)" }]} />
          <div className="mt-3 rounded-lg border border-line bg-surface-2 px-3 py-2 font-mono text-[0.75rem] text-muted" aria-live="polite">
            {last ? (
              <>
                <div>at {rc(last.s)} took {ARROW[last.a]} → {rc(last.s2)}, reward {last.r}{last.done ? " (episode ends)" : ""}</div>
                <div>target = {last.done ? `${last.r} (terminal, no bootstrap)` : `${last.r} + ${fmt(last.gamma, 2)} × ${fmt(last.boot, 2)} = ${fmt(last.target, 2)}`}{!last.done && <span className="text-faint"> ({p.algo === "q" ? "max next Q" : "Q of next action"})</span>}</div>
                <div>Q ← {fmt(last.old, 2)} + {fmt(last.alpha, 2)} × ({fmt(last.target, 2)} − {fmt(last.old, 2)}) = <span className="text-ink">{fmt(last.next, 2)}</span></div>
              </>
            ) : <div>{ALGO_NAME[p.algo]}: press Step to see one update written out.</div>}
          </div>
        </div>
      </div>
    </LabFrame>
  );
}
