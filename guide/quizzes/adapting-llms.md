<!-- GENERATED from adapting-llms.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Adapting LLMs

Covers the lesson [Adapting LLMs](../lessons/15-llms/03-adapting-llms.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/adapting-llms/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

LoRA with rank $r = 4$ on a $1024 \times 1024$ matrix. How many parameters are trained?

<details>
<summary>Answer</summary>

**8192**

$4 \times (1024 + 1024) = 8192$, versus 1,048,576 for the full matrix (0.78%).

</details>

## 2. Match (medium)

Match each technique to what it changes.

| Concept | Options |
|---|---|
| Few-shot prompting | All of the model's weights |
| RAG | A small set of added weights |
| LoRA | The context at runtime, with retrieved documents |
| Full fine-tuning | The context at runtime |

<details>
<summary>Answer</summary>

- Few-shot prompting → The context at runtime
- RAG → The context at runtime, with retrieved documents
- LoRA → A small set of added weights
- Full fine-tuning → All of the model's weights

Context-side techniques need no training; weight-side techniques do.

</details>

## 3. Multiple choice (medium)

A support bot must answer using the current refund policy, which changes monthly. What is the best primary approach?

- **A.** Fine-tune monthly on the new policy.
- **B.** Retrieve the current policy at query time (RAG) and cite it.
- **C.** Raise the temperature.
- **D.** Pretrain a new model.

<details>
<summary>Answer</summary>

**B.** Retrieve the current policy at query time (RAG) and cite it.

Frequently changing facts belong in context, where updating is a re-index, and citations make answers checkable.

</details>

## 4. Multiple choice (medium)

Despite careful prompting, a model keeps drifting from your brand voice and output structure across thousands of requests. What is a reasonable next step?

- **A.** Add more documents to the RAG index.
- **B.** Parameter-efficient fine-tuning (e.g. LoRA) on high-quality examples of the desired voice and structure.
- **C.** Lower top-k to 1.
- **D.** Switch to a smaller context window.

<details>
<summary>Answer</summary>

**B.** Parameter-efficient fine-tuning (e.g. LoRA) on high-quality examples of the desired voice and structure.

Voice and format are behaviors; fine-tuning shapes behavior. Evaluate before and after.

</details>

## 5. Select all that apply (medium)

Which measures meaningfully reduce harmful hallucinations in a Q&A product? Select all that apply.

- **A.** Grounding answers in retrieved sources with citations
- **B.** Letting the model say "I don't know" when sources lack the answer
- **C.** Setting temperature to 0 and assuming answers are now correct
- **D.** Verifying claims against sources and human review for high-stakes answers

<details>
<summary>Answer</summary>

**A, B, D**

Temperature affects randomness, not truthfulness.

</details>

## 6. Reflection (hard)

In one or two sentences, what does preference optimization (RLHF or DPO) train on, and what does it change about the model?

<details>
<summary>Answer</summary>

**Model answer.** It trains on prompts paired with responses ranked by humans (or a model), such as a preferred and a rejected answer, and adjusts the weights so the model assigns higher probability to preferred-style responses. It mainly shapes behavior (helpfulness, harmlessness, tone, refusals) rather than adding knowledge.

Preference data shapes how the model responds, not what facts it knows.

</details>
