"use client";
// Guided tours for labs. A tour is a list of steps; each step puts the lab into a state and
// (optionally) animates one value smoothly, with a caption explaining what you're seeing.
// The same steps drive two things: Watch mode inside LabFrame, and scrolling explainers,
// where the paragraph in view picks the step (via TourContext).
import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState } from "react";
import { CaretLeft, CaretRight, Pause, Play, SpeakerHigh, SpeakerSlash, X } from "@phosphor-icons/react";

export interface TourStep {
  id: string;                     // explainers reference steps by id
  caption: string;                // 1–2 plain sentences (read aloud when narration is on)
  apply?: () => void;             // set the lab's state. Steps are absolute: set everything this step depends on.
  animate?: (t: number) => void;  // optional: called every frame with eased t from 0 to 1, after apply
  animMs?: number;                // animation length (default 1800)
  holdMs?: number;                // Watch mode: time to stay on the step after animating (default from caption length)
}

/** Set by a scrolling explainer: the step id that matches the paragraph in view. */
export const TourContext = createContext<{ step: string | null } | null>(null);

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const reduceMotion = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const readMs = (caption: string) => Math.max(3500, caption.split(/\s+/).length * 330);

/** Runs one step: apply, then animate over animMs. Always reads the latest tour, so closures stay fresh. */
function useStepRunner(tour: TourStep[] | undefined) {
  const tourRef = useRef(tour);
  useLayoutEffect(() => { tourRef.current = tour; }); // before the next frame reads it
  const raf = useRef(0);
  const cancel = () => cancelAnimationFrame(raf.current);
  const run = (i: number, onDone?: () => void) => {
    cancel();
    const s = tourRef.current?.[i];
    if (!s) return;
    s.apply?.();
    if (!s.animate) { onDone?.(); return; }
    const dur = reduceMotion() ? 0 : (s.animMs ?? 1800);
    const t0 = performance.now();
    const frame = (now: number) => {
      const t = dur ? Math.min(1, (now - t0) / dur) : 1;
      tourRef.current?.[i]?.animate?.(ease(t));
      if (t < 1) raf.current = requestAnimationFrame(frame);
      else onDone?.();
    };
    // wait one frame so state set by apply has rendered before the first animate call
    raf.current = requestAnimationFrame(() => { raf.current = requestAnimationFrame(frame); });
  };
  useEffect(() => cancel, []);
  return { run, cancel };
}

/** Scrolling explainers: when the paragraph in view changes, jump the lab to that step. */
export function useScrollDrivenTour(tour: TourStep[] | undefined) {
  const ctx = useContext(TourContext);
  const { run } = useStepRunner(tour);
  const step = ctx?.step;
  useEffect(() => {
    if (!step) return;
    const i = tour?.findIndex((s) => s.id === step) ?? -1;
    if (i >= 0) run(i);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- re-run only when the step in view changes
  }, [step]);
  return !!ctx;
}

const iconBtn = "rounded-md p-1.5 text-muted hover:text-ink hover:bg-surface-2 disabled:opacity-30";

/** Watch mode player: plays the steps in order with captions and optional narration. */
export function TourPlayer({ tour, onExit }: { tour: TourStep[]; onExit: () => void }) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [voice, setVoice] = useState(false);
  // Index of the step whose animation / narration has finished (so pause, play, and the voice
  // toggle never restart a step: only changing the step does).
  const [animatedFor, setAnimatedFor] = useState(-1);
  const [spokenFor, setSpokenFor] = useState(-1);
  const { run, cancel } = useStepRunner(tour);
  const s = tour[i];
  const synth = typeof window !== "undefined" ? window.speechSynthesis : undefined;

  // 1. Apply and animate a step exactly once, when it becomes current.
  useEffect(() => {
    run(i, () => setAnimatedFor(i));
    return cancel;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only a new step re-runs its animation
  }, [i]);

  // 2. Narration: speaks the current caption once while playing with voice on. Pause, voice off,
  //    exit, or a new step all silence it immediately.
  useEffect(() => {
    if (!synth) return;
    synth.cancel();
    if (!voice || !playing || spokenFor === i) return;
    let live = true; // cancel() fires "end" on the old utterance; ignore it
    const u = new SpeechSynthesisUtterance(s.caption);
    u.rate = 1.02;
    u.onend = () => { if (live) setSpokenFor(i); };
    synth.speak(u);
    return () => { live = false; synth.cancel(); };
  }, [i, voice, playing, spokenFor, s.caption, synth]);

  // 3. Auto-advance once the step has finished animating (and speaking), unless paused.
  useEffect(() => {
    if (!playing || animatedFor !== i || (voice && synth && spokenFor !== i)) return;
    const t = setTimeout(() => (i + 1 < tour.length ? setI(i + 1) : setPlaying(false)), voice ? 700 : (s.holdMs ?? readMs(s.caption)));
    return () => clearTimeout(t);
  }, [i, playing, voice, animatedFor, spokenFor, synth, tour.length, s]);

  const go = (n: number) => { setI(Math.max(0, Math.min(tour.length - 1, n))); };
  const togglePlay = () => {
    if (!playing && i === tour.length - 1 && animatedFor === i) setI(0); // replay from the start
    setPlaying(!playing);
  };

  return (
    <div className="border-b border-line bg-accent-soft/50 px-4 py-3 sm:px-5" role="region" aria-label="Guided tour" aria-live="polite">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 shrink-0 rounded-full bg-accent px-2 py-0.5 font-mono text-[0.7rem] text-white dark:text-zinc-950">{i + 1}/{tour.length}</span>
        <p className="flex-1 text-[0.95rem] leading-relaxed text-ink">{s.caption}</p>
        <div className="flex shrink-0 items-center gap-0.5">
          <button type="button" className={iconBtn} onClick={() => go(i - 1)} disabled={i === 0} aria-label="Previous step"><CaretLeft size={16} /></button>
          <button type="button" className={iconBtn} onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={16} /> : <Play size={16} />}</button>
          <button type="button" className={iconBtn} onClick={() => go(i + 1)} disabled={i === tour.length - 1} aria-label="Next step"><CaretRight size={16} /></button>
          <button type="button" className={iconBtn} onClick={() => setVoice(!voice)} aria-pressed={voice} aria-label={voice ? "Turn narration off" : "Turn narration on"}>{voice ? <SpeakerHigh size={16} /> : <SpeakerSlash size={16} />}</button>
          <button type="button" className={iconBtn} onClick={onExit} aria-label="Exit tour"><X size={16} /></button>
        </div>
      </div>
      <div className="mt-2 flex gap-1" aria-hidden>
        {tour.map((_, k) => <span key={k} className={`h-1 flex-1 rounded-full ${k <= i ? "bg-accent" : "bg-line"}`} />)}
      </div>
      {!playing && i === tour.length - 1 && <p className="mt-2 text-xs text-muted">Tour finished. The controls are yours: change something and predict what happens first.</p>}
    </div>
  );
}
