# ML to LLMs

A visual, math-first course that takes a software engineer from linear regression to transformers, RAG, and production AI engineering. Every concept is taught in six layers: **intuition, visualization, math (with every arithmetic step), implementation, engineering, and a knowledge check.**

The same content ships two ways:

| | For | Where |
|---|---|---|
| **Website** | learning interactively: 20 labs, a Math Lab with 18 step-by-step calculators, quizzes with a review queue, and local progress tracking | [`web/`](web/) → **[sreshtalluri.github.io/ml-ai-learn](https://sreshtalluri.github.io/ml-ai-learn/)** |
| **GitHub edition** | reading on GitHub with no build step: markdown lessons, committed figures, runnable Python, quizzes with hidden answers | [`guide/`](guide/) → **[start here](guide/README.md)** |

There is one source of truth. Lessons, quizzes, the glossary, model cards, and cheat sheets live in `guide/`, and the website renders those same files at build time. Fixing a typo in a lesson fixes it in both places.

## What's inside

- **21 modules, 41 lessons**: math foundations, ML vocabulary and workflow, regression, classification, KNN, Naive Bayes, SVMs, trees and boosting, clustering, PCA, classical NLP and embeddings, neural networks, gradient descent, backpropagation, training and regularization, CNNs, RNNs, autoencoders and diffusion, self-attention, transformers, tokenization and pretraining, decoding, adapting LLMs (LoRA, RLHF/DPO), RAG, LLM evaluation, production architecture, reliability and cost, AI security, a 12-week plan, and a five-project portfolio ladder.
- **Worked examples verified by code.** Every hand calculation in a lesson is reproduced by a script in [`guide/code/`](guide/code/) and by unit tests in `web/`.
- **20 interactive labs**, each with predict-then-check "Try it" prompts: learning-paradigm guide, underfitting vs overfitting, linear regression (drag points, L1/L2), classification threshold (costs, ROC, calibration), KNN (scaling), decision-tree builder (Gini splits), gradient boosting round by round, K-means, PCA projection, TF-IDF, neural-network forward pass, gradient descent (SGD/momentum/Adam, divergence), backpropagation (nine steps), convolution and pooling, self-attention (Q, K, V to output with shapes), BPE tokenizer with a context-window meter, decoding (temperature, top-k, top-p), a RAG pipeline (chunking, retrieval, reranking, misses), a clickable production architecture with request tracing, and a prompt-injection lab with mocked tools.
- **Learning paths**: the full 12-week course, a 4-week AI-engineering fast track, and an interview-prep pass ([guide/learning-paths.md](guide/learning-paths.md)).
- **41 quizzes** with seven question types (multiple choice, multi-select, calculation, fill-in, ordering, matching, reflection), explanations for every option, and a review queue.
- **28 model cards**, **18 cheat sheets**, and a **49-term glossary** (with hover definitions inside lessons on the website).

## Quick start

```bash
# Website
cd web
npm ci
npm run dev                 # http://localhost:3000

# GitHub edition's runnable code (installs numpy, matplotlib, scikit-learn on first run)
cd guide
uv run code/03-regression/linear_regression.py
```

No paid APIs, no accounts, no backend. Progress is stored in your browser's local storage, and you can export and import it as JSON.

## Architecture

```
guide/                         single source of truth (GitHub-readable markdown)
  lessons/NN-module/           module README + one file per lesson (frontmatter + six sections)
  quizzes/*.yml                canonical quizzes → generated *.md views for GitHub readers
  models/, cheatsheets/        model cards and reference pages
  glossary.md                  parsed into the searchable glossary and hover tooltips
  code/, figures/              runnable Python scripts and the figures they produce
  AUTHORING.md                 the content format (read this before contributing)
web/                           Next.js 16 (App Router, TypeScript strict, Tailwind v4), static export
  src/lib/content.ts           reads ../guide at build time; maps .md links to site routes
  src/components/Markdown.tsx  GitHub-flavored markdown + KaTeX; swaps lab/quiz markers for live components
  src/components/labs/         one component per lab, shared LabFrame/Plot primitives
  src/lib/ml.ts, optim.ts      pure math used by labs and the Math Lab (unit-tested)
  src/lib/calculators/         Math Lab: each calculator is data + a pure compute() returning steps
  src/lib/progress.ts          local-first progress store (useSyncExternalStore + localStorage)
docs/                          the original course spec and the v1 source guide
```

**How one file serves both audiences.** Lessons embed invisible HTML-comment markers:

```markdown
<!-- lab:linear-regression -->
![static figure for GitHub readers](../../figures/linear-regression.png)
<!-- /lab -->
```

GitHub shows the figure. The website replaces everything between the markers with the interactive lab. Quizzes work the same way. Relative links to `.md` files become site routes, links to `.py` files open on GitHub, and links to `glossary.md#term` become hover definitions.

**Key decisions** (documented trade-offs):
- *Static export, no backend.* The site deploys to GitHub Pages and runs offline after loading. Cloud sync or an AI tutor can be added behind the `progress.ts` interface later.
- *Hand-built SVG labs instead of a chart library.* Each visualization is a small React component over shared `Plot`/`LabFrame` primitives, which keeps bundles small and every number inspectable.
- *Plain Tailwind instead of shadcn/ui.* The component surface is small (buttons, sliders, segmented controls), so a component library wasn't worth the dependency.
- *Labs fall back to static figures.* Every lab has a committed figure generated by Python, so the GitHub edition loses interactivity, not content.

## Quality checks

```bash
cd web
npm test               # 54 tests: math, worked examples, progress persistence, labs, calculators, content integrity
npm run typecheck
npm run lint
npm run check:quizzes  # quiz YAML valid and generated markdown up to date
npm run build          # static site in web/out
```

The content-integrity tests fail the build if any relative link in `guide/` is broken, a lesson references a missing prerequisite or lab, a lesson is missing one of its six sections, or a glossary anchor doesn't exist.

## Deployment

`.github/workflows/site.yml` runs every check on pull requests. On pushes to `main` it builds with `NEXT_PUBLIC_BASE_PATH=/ml-ai-learn` and deploys `web/out` to GitHub Pages.

## Roadmap

See [`docs/COURSE_DESIGN.md`](docs/COURSE_DESIGN.md) for the lesson-by-lesson review that decides which labs to build, and [`docs/BACKLOG.md`](docs/BACKLOG.md) for what's next.

## Credits

The curriculum started from a v1 study guide (kept in [`docs/source-guide-v1.pdf`](docs/source-guide-v1.pdf)), reorganized and expanded according to the course spec in [`docs/PROMPT_SPEC.md`](docs/PROMPT_SPEC.md). All datasets are synthetic and generated with fixed seeds; no benchmark numbers are invented.
