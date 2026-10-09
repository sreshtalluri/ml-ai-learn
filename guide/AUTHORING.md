# Authoring guide

Everything in `guide/` is the **single source of truth**. GitHub renders it directly for readers, and the website in `web/` reads the same files at build time. Write once and both audiences see it.

## Layout

```
guide/
  README.md                         course index (GitHub entry point)
  glossary.md                       all glossary terms
  lessons/NN-module-slug/
    README.md                       module overview (frontmatter: title, summary, skill)
    NN-lesson-slug.md               one lesson per file (frontmatter below)
  quizzes/<lesson-slug>.yml         quiz source (canonical)
  quizzes/<lesson-slug>.md          GENERATED from the .yml (npm run gen:quizzes in web/)
  models/<model-id>.md              model cards for the Model Explorer
  cheatsheets/<id>.md               concise reference pages (never name one agents.md: tools read AGENTS.md as instructions)
  sprints/<role>.md                 7-day interview sprints (frontmatter: title, role, order, summary; `## Day 1:` .. `## Day 7:`)
  code/NN-module-slug/<name>.py     runnable Python (from scratch + library)
  figures/<lesson-slug>*.png        figures produced by the Python scripts
```

The lesson slug is the file name without the `NN-` prefix and `.md`. Slugs must be unique across the course.

## Lesson frontmatter

```yaml
---
title: Linear regression
summary: One sentence that says what the learner will be able to do.
skill: classical-ml   # math | classical-ml | nlp | deep-learning | transformers | llms | engineering
minutes: 35
prerequisites: [vectors-and-matrices, calculus-for-ml]   # lesson slugs
related: [regularization, gradient-descent]               # lesson slugs
---
```

## Lesson body (the six layers)

Every lesson follows this skeleton. Keep the numbered `##` headings, since the website builds its table of contents from `##` headings.

```markdown
# Title

> **Mental model.** One or two sentences, plain English, with an analogy.

**You will learn to**
- objective (a verb the learner can be tested on)

**Why it matters.** One short paragraph.

## 1. Intuition
## 2. Visualization
## 3. The math
## 4. Implementation
## 5. Engineering
## 6. Knowledge check
## Summary
```

Section rules:

- **Intuition.** Plain English first. No symbols until they are defined.
- **Visualization.** Show a figure (PNG from `figures/`). If the site has an interactive lab for it, wrap the static figure in lab markers (see below).
- **The math.** Include a symbol table (`| Symbol | Meaning | Shape |`), then the formula, then a worked numerical example where every arithmetic step is written out. Never write "it works out".
- **Implementation.** Give a from-scratch NumPy version and a library version (scikit-learn / PyTorch). Keep inline snippets short and link the runnable script in `code/`.
- **Engineering.** Cover use cases, preprocessing, compute cost, production concerns, and failure modes, plus a `### Common mistakes` subsection.
- **Knowledge check.** Wrap a link to the quiz in quiz markers, then add one practice exercise and one small implementation challenge, with a solution in `<details>`.
- **Summary.** 3 to 6 bullets, then `**Next:**` and `**Related:**` links.
- **Interview angle** (last section, `## Interview angle`). 3 to 5 questions the way strong interviewers ask them: one on the mechanism, one trade-off, one failure to diagnose, and one quick calculation or design follow-up. Each answer is 100 to 170 words, something you could say out loud in a minute. The website's rapid-fire drill parses this exact shape:

  ```markdown
  <details>
  <summary><strong>Why is attention scaled by the square root of d_k?</strong></summary>

  The answer, in markdown. Math is fine here.

  </details>
  ```

  The question sits inside raw HTML, so it can't contain `$math$` or backslash escapes (a test enforces this). Write symbols as plain text or Unicode: "d_k", "σ²", "USD 500".

**Quick read.** The website's quick-read switch hides sections 3 to 6 and keeps the intro, Intuition, Visualization, Summary, and Interview angle. Write Intuition and Visualization so they stand on their own without the math.

## Math

- Inline math uses `$...$`. Avoid `$` for currency; write "USD 50k" or "\$" outside math.
- Display math uses a fenced block with the language `math`:

  ````markdown
  ```math
  \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  ```
  ````

Both GitHub and the website render these forms. Do not use `$$`. Never put math in headings (it breaks anchors and the table of contents).

Write currency as words or with an escaped dollar sign (`\$290k`), otherwise `$` starts math mode.

## Callouts

Use GitHub alert syntax. The website styles these the same way:

```markdown
> [!NOTE]
> Intuition or extra context.

> [!TIP]
> Engineering note.

> [!WARNING]
> Common mistake or failure mode.

> [!IMPORTANT]
> Key distinction (e.g. RAG vs fine-tuning).
```

## Interactive markers (invisible on GitHub)

```markdown
<!-- lab:linear-regression -->
![Residuals for a synthetic dataset](../../figures/linear-regression.png)

*Interactive version: [open the lab on the website](https://sreshtalluri.github.io/ml-ai-learn/labs/linear-regression/).*
<!-- /lab -->
```

On GitHub the reader sees the static figure and the link. On the website, everything between the markers is replaced by the interactive lab with that id.

```markdown
<!-- quiz:linear-regression -->
**[Take the linear regression quiz](../../quizzes/linear-regression.md)**
<!-- /quiz -->
```

On the website this becomes the interactive quiz.

## Visual explainers and guided tours

Every lab has a **guided tour**: 4 to 8 steps, defined in the lab component as a `tour` prop on `LabFrame` (see `GradientDescentLab.tsx`). Each step has:

- `id`: short kebab-case name, used by explainers.
- `caption`: one or two plain sentences (no math markup; it's read aloud when narration is on). Say what to look at and why it matters, like a narrator pointing at the screen.
- `apply`: puts the lab into the state for this step. Steps are absolute: set every control the step depends on, so any step works when jumped to directly.
- `animate` (optional): called every frame with `t` from 0 to 1 after `apply`. Use it to move one value smoothly (a slider sweeping, an optimizer walking, a kernel sliding) so the learner sees continuous change rather than a cut.

The website's **Watch** button plays the tour with captions, optional narration, and pause, back, and next controls.

A **visual explainer** (`explainers/<id>.md`, frontmatter `title, summary, lab, lesson, minutes`) is the long-form version: short sections of prose, each starting with `<!-- step:<id> -->`. On the website the lab is pinned beside the text and switches to that step when the section scrolls into view. On GitHub the markers are invisible, so put the lab's static figure in lab markers at the top and keep each section readable on its own. Aim for 5 to 8 sections of 60 to 150 words: one idea per section, in the same order as the tour.

## Links

Always link with **relative paths to `.md` files**. The website rewrites them to its own routes. Link glossary terms as `[logit](../../glossary.md#logit)`, which the website shows with a hover definition. Links to `.py` files open on GitHub.

## Quizzes (`quizzes/<slug>.yml`)

```yaml
title: Linear regression
lesson: linear-regression
questions:
  - id: lr-mse-calc                 # unique within the file
    type: numeric                   # single | multi | numeric | fill | order | match | short
    difficulty: medium              # easy | medium | hard
    prompt: Markdown text. Math allowed.
    answer: 2.5                     # numeric: number; single: option index (0-based); multi: [indices];
                                    # fill: [accepted strings, case-insensitive]; order/match/short: omit
    tolerance: 0.01                 # numeric only
    options: [..]                   # single, multi, order (order: list them in the CORRECT order)
    pairs: [[left, right], ...]     # match (listed correctly paired)
    explanation: Why the answer is correct, with working.
    why_not: [per-option text]      # optional, single/multi: one entry per option
    sample_answer: ...              # short only (self-graded reflection)
```

Every quiz needs at least 6 questions, and no more than half may be pure recall. Include derivation (`numeric`), diagnosis (a debugging scenario), and comparison or design questions.

After editing YAML, run `npm run gen:quizzes` in `web/` to regenerate the markdown view.

## Glossary (`glossary.md`)

Each term is a `###` heading followed by labelled lines (labels are parsed by the site). Separate lines with a blank line.

```markdown
### Logit

**Plain English:** The raw score a model outputs before it is turned into a probability.

**Formal:** For logistic regression, $z = w \cdot x + b$; the probability is $\sigma(z)$.

**Example:** A logit of 0 corresponds to probability 0.5.

**Related:** [Sigmoid](#sigmoid), [Softmax](#softmax)

**Lesson:** [Logistic regression](lessons/04-classification/01-logistic-regression.md)
```

## Model cards (`models/<id>.md`)

```yaml
---
name: Random forest
tags: [supervised, classification, regression, interpretable, small-data]
lessons: [trees-and-ensembles]   # lesson slugs
labs: []                         # lab ids
---
```

Allowed tags: supervised, unsupervised, self-supervised, regression, classification, clustering, dimensionality-reduction, nlp, vision, generative, interpretable, low-latency, small-data.

Body sections, as `##` headings in this order: Problem type, Input, Output, Mental model, Core objective, Training process, Preprocessing, Assumptions, Key hyperparameters, Good use cases, Poor use cases, Strengths, Weaknesses, Computational cost, Evaluation metrics, Failure modes, Minimal implementation, Compared with neighbors, Learn more.

## Python code and figures

All scripts live in `code/` and run with [uv](https://docs.astral.sh/uv/) from inside `guide/`:

```bash
uv run code/03-regression/linear_regression.py
```

Scripts must be deterministic (fixed seeds, synthetic data labelled as synthetic), print their worked example, and save figures to `figures/` with the `Agg` backend. Use `code/_style.py` for consistent figure styling. Figures are committed so GitHub readers see them without running anything.

## Pedagogical rules

- Start with intuition before equations. Define every symbol. Write tensor shapes.
- Show every arithmetic step. Compare against a simple baseline.
- Distinguish: loss vs metric, parameter vs hyperparameter, training vs inference, RAG vs fine-tuning, model confidence vs factual certainty, correlation vs causation, discovered cluster vs real category.
- Label synthetic data as synthetic. Never invent benchmark numbers.
