"use client";
import { useState } from "react";
import { LabFrame } from "./ui";

type Answer = string | null;
interface Q { id: string; text: string; options: { value: string; label: string }[] }

const QUESTIONS: Q[] = [
  { id: "signal", text: "What feedback does the data give you?", options: [
    { value: "labels", label: "A known correct answer for each example (labels)" },
    { value: "some", label: "Labels for a few examples, many unlabeled" },
    { value: "none", label: "No answers at all, just the inputs" },
    { value: "self", label: "The answer is hidden inside the data itself (next word, masked pixel)" },
    { value: "reward", label: "A reward after actions, often delayed" },
  ] },
  { id: "output", text: "What should the system produce?", options: [
    { value: "number", label: "A number (price, demand, duration)" },
    { value: "category", label: "A category or yes/no" },
    { value: "structure", label: "Groups, a compressed view, or unusual points" },
    { value: "representation", label: "A general-purpose representation or generator" },
    { value: "actions", label: "A sequence of decisions" },
  ] },
  { id: "time", text: "Is there an ordering in time that matters?", options: [
    { value: "yes", label: "Yes, the future must never leak into training" },
    { value: "no", label: "No, examples are independent" },
  ] },
];

function recommend(a: Record<string, Answer>) {
  const { signal, output, time } = a;
  if (!signal) return null;
  const timeNote = time === "yes" ? " Split by time, not randomly, so validation mimics predicting the future." : "";
  if (signal === "reward" || output === "actions")
    return { name: "Reinforcement learning", why: "The only feedback is a reward that depends on a sequence of actions, so the model must learn a policy by trial and error. In practice, frame it as supervised learning first if you have logged decisions with known outcomes; RL is expensive and unstable." + timeNote };
  if (signal === "self")
    return { name: "Self-supervised learning", why: "The labels are created from the data itself: hide part of the input and predict it. This is how LLMs are pretrained (next-token prediction) and how embedding models learn representations." + timeNote };
  if (signal === "some")
    return { name: "Semi-supervised learning", why: "Train on the few labels, use the model to pseudo-label confident unlabeled examples, and retrain; or pretrain a representation on all the data and fine-tune on the labels. Check that pseudo-labels don't amplify the model's own mistakes." + timeNote };
  if (signal === "none" || output === "structure")
    return { name: "Unsupervised learning", why: "With no target, the goal is structure: clustering (K-means, DBSCAN), dimensionality reduction (PCA), or anomaly detection. A discovered cluster is a hypothesis, not a verified real-world category, so review clusters with domain experts." + timeNote };
  if (output === "number")
    return { name: "Supervised learning: regression", why: "Labeled examples with a continuous target. Start with linear or ridge regression as the baseline, then try gradient-boosted trees. Report MAE or RMSE in the target's units." + timeNote };
  return { name: "Supervised learning: classification", why: "Labeled examples with a categorical target. Start with logistic regression, then gradient boosting. Choose the decision threshold from the real cost of false positives versus false negatives." + timeNote };
}

const SCENARIOS: { title: string; answers: Record<string, string>; teaching: string }[] = [
  { title: "House-price prediction", answers: { signal: "labels", output: "number", time: "no" }, teaching: "Past sales give a price for every house: a textbook regression target." },
  { title: "Fraud detection", answers: { signal: "labels", output: "category", time: "yes" }, teaching: "Confirmed fraud cases are labels, but they are rare (class imbalance) and arrive late. Use precision-recall metrics and a time split." },
  { title: "Customer segmentation", answers: { signal: "none", output: "structure", time: "no" }, teaching: "No one has labeled customers by segment. Cluster on behavior, then let the business decide whether the clusters mean anything." },
  { title: "Anomaly detection in server logs", answers: { signal: "none", output: "structure", time: "yes" }, teaching: "Incidents are rare and unlabeled, so model normal behavior and flag outliers. If you later collect labeled incidents, switch to supervised." },
  { title: "Next-token prediction (LLM pretraining)", answers: { signal: "self", output: "representation", time: "no" }, teaching: "Every position in a text is a free label: the next token. That is why LLMs can train on trillions of tokens with no human labeling." },
  { title: "Product recommendation", answers: { signal: "some", output: "category", time: "yes" }, teaching: "Clicks and purchases are implicit, partial labels: a non-click is not a confirmed dislike. Many systems combine self-supervised embeddings with supervised ranking." },
  { title: "Robot arm control", answers: { signal: "reward", output: "actions", time: "yes" }, teaching: "Success is measured after a sequence of motor commands. Reinforcement learning, often bootstrapped by imitation of demonstrations." },
];

export default function ParadigmGuideLab() {
  const [answers, setAnswers] = useState<Record<string, Answer>>({ signal: null, output: null, time: null });
  const [scenario, setScenario] = useState<number | null>(null);
  const rec = recommend(answers);

  return (
    <LabFrame
      id="paradigm-guide"
      title="Which learning paradigm fits?"
      subtitle="Answer three questions, or load one of the seven scenarios."
      onReset={() => { setAnswers({ signal: null, output: null, time: null }); setScenario(null); }}
      interpretation={rec ? <><p className="text-ink font-medium">{rec.name}</p><p className="mt-1">{rec.why}</p>{scenario !== null && <p className="mt-2 text-ink">{SCENARIOS[scenario].teaching}</p>}</> : "Start with the first question: the kind of feedback in your data decides the paradigm more than anything else."}
    >
      <div className="grid gap-6 @3xl:grid-cols-[minmax(0,1fr)_14rem]">
        <div className="space-y-5">
          {QUESTIONS.map((q, qi) => (
            <fieldset key={q.id}>
              <legend className="text-sm font-medium mb-2"><span className="font-mono text-faint mr-2">{qi + 1}</span>{q.text}</legend>
              <div className="grid gap-1.5">
                {q.options.map((o) => (
                  <button key={o.value} type="button" aria-pressed={answers[q.id] === o.value}
                    onClick={() => { setAnswers({ ...answers, [q.id]: o.value }); setScenario(null); }}
                    className={`text-left rounded-lg border px-3 py-2 text-sm ${answers[q.id] === o.value ? "border-accent bg-accent-soft" : "border-line hover:border-faint"}`}>
                    {o.label}
                  </button>
                ))}
              </div>
            </fieldset>
          ))}
        </div>
        <div>
          <p className="text-sm font-medium mb-2">Scenarios</p>
          <div className="grid gap-1.5">
            {SCENARIOS.map((s, i) => (
              <button key={s.title} type="button" aria-pressed={scenario === i} onClick={() => { setAnswers(s.answers); setScenario(i); }}
                className={`text-left rounded-lg border px-3 py-2 text-sm ${scenario === i ? "border-accent bg-accent-soft" : "border-line hover:border-faint text-muted"}`}>
                {s.title}
              </button>
            ))}
          </div>
        </div>
      </div>
    </LabFrame>
  );
}
