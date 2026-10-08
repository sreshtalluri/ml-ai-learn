export const SKILLS = {
  math: { label: "Mathematics", color: "var(--c-blue)" },
  "classical-ml": { label: "Classical ML", color: "var(--c-teal)" },
  nlp: { label: "NLP", color: "var(--c-orange)" },
  "deep-learning": { label: "Deep learning", color: "var(--c-purple)" },
  transformers: { label: "Transformers", color: "var(--c-blue)" },
  llms: { label: "LLMs", color: "var(--c-teal)" },
  engineering: { label: "AI engineering", color: "var(--c-orange)" },
} as const;

export type Skill = keyof typeof SKILLS;
export const SKILL_KEYS = Object.keys(SKILLS) as Skill[];
