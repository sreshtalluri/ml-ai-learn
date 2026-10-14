<!-- GENERATED from context-engineering.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Context engineering

Covers the lesson [Context engineering](../lessons/28-prompting-and-context/03-context-engineering.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/context-engineering/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

How does context engineering differ from prompt engineering?

- **A.** They are the same thing
- **B.** Prompt engineering is the wording of instructions; context engineering is everything in the window (instructions, tools, retrieved documents, history, memory) and its order
- **C.** Context engineering is database design for embeddings
- **D.** Prompt engineering is the more advanced discipline

<details>
<summary>Answer</summary>

**B.** Prompt engineering is the wording of instructions; context engineering is everything in the window (instructions, tools, retrieved documents, history, memory) and its order

The prompt is one part of the context. In RAG and agents, the dynamically assembled rest of the window usually matters more.

- **A:** They overlap only in a single-prompt chatbot.
- **B:** Correct.
- **C:** Vector databases feed the context, but context engineering is about what reaches the model.
- **D:** Neither is "more advanced"; context engineering is the broader scope.

</details>

## 2. Multiple choice (easy)

Why must you reserve output tokens when budgeting a context window?

- **A.** Providers charge a fixed fee for the reserve
- **B.** The window limit covers input and output together, so a full input leaves no room for the answer
- **C.** The reserve stores the KV cache
- **D.** Output tokens are free

<details>
<summary>Answer</summary>

**B.** The window limit covers input and output together, so a full input leaves no room for the answer

Input plus output must fit in the window. Without a reserve, long inputs truncate the answer.

- **A:** There is no reserve fee; output tokens are billed as generated.
- **B:** Correct.
- **C:** The KV cache lives in server memory, not in a token reserve.
- **D:** Output tokens usually cost more than input tokens.

</details>

## 3. Multiple choice (medium)

You must pass 10 retrieved chunks to a model. Where should the two most relevant go?

- **A.** Both in the middle, surrounded by context
- **B.** At the edges of the retrieved block (first and last), with the weakest chunks in the middle
- **C.** Alphabetical order
- **D.** Random order every call

<details>
<summary>Answer</summary>

**B.** At the edges of the retrieved block (first and last), with the weakest chunks in the middle

Models often use information near the start and end of long inputs more reliably than information in the middle.

- **A:** The middle is where facts are most often underused.
- **B:** Correct.
- **C:** Alphabetical order ignores relevance.
- **D:** Random order makes results unstable and also breaks prefix caching.

</details>

## 4. Calculation (medium)

Fixed context is 5,000 tokens and each turn adds 300 tokens of history, kept in full. How many input tokens are billed over 10 calls?

<details>
<summary>Answer</summary>

**63500**

$10 \times 5{,}000 + 300 \times \frac{10 \times 9}{2} = 50{,}000 + 300 \times 45 = 50{,}000 + 13{,}500 = 63{,}500$.

</details>

## 5. Calculation (hard)

Same setup (5,000 fixed, 300 per turn), but history is compacted to the last 2 turns plus a 100-token summary once there are more than 2 earlier turns. What is the input for call 10?

<details>
<summary>Answer</summary>

**5700**

Call 10 has 9 earlier turns, more than 2, so history is $2 \times 300 + 100 = 700$. Input $= 5{,}000 + 700 = 5{,}700$, versus $5{,}000 + 2{,}700 = 7{,}700$ with full history.

</details>

## 6. Multiple choice (medium)

Prefix caching stopped working after a release. Which change most likely caused it?

- **A.** Retrieved documents are now appended after the user query
- **B.** The current timestamp was added to the first line of the system prompt
- **C.** The output reserve was increased
- **D.** The temperature was lowered

<details>
<summary>Answer</summary>

**B.** The current timestamp was added to the first line of the system prompt

Caches match byte-identical prefixes. A changing value at the very top makes every prefix unique.

- **A:** Content after the stable prefix doesn't affect prefix reuse (though query-last is usually better).
- **B:** Correct.
- **C:** Output size doesn't change the input prefix.
- **D:** Sampling settings don't change the input tokens.

</details>

## 7. Multiple choice (medium)

An agent's answers degrade in long sessions. Logs show the context is 80 percent raw tool output from earlier steps. What is the best fix?

- **A.** Switch to a model with a larger window
- **B.** Trim tool outputs to needed fields, summarize old turns, keep a structured notes block of decisions, and let the agent re-fetch old results by ID
- **C.** Delete the system prompt to save space
- **D.** Raise the temperature so the agent explores more

<details>
<summary>Answer</summary>

**B.** Trim tool outputs to needed fields, summarize old turns, keep a structured notes block of decisions, and let the agent re-fetch old results by ID

The problem is noise crowding out instructions and key facts. Compaction and selection fix the cause.

- **A:** A bigger window holds more noise; cost rises and quality need not improve.
- **B:** Correct.
- **C:** The system prompt holds the rules the agent is forgetting.
- **D:** Randomness doesn't remove the noise.

</details>

## 8. Multiple choice (medium)

A user's preferred programming language should persist across sessions. Where should it live?

- **A.** Only in the current conversation history
- **B.** In a long-term memory store, retrieved or loaded into the context at session start
- **C.** In the model's weights, via a nightly fine-tune
- **D.** In the output reserve

<details>
<summary>Answer</summary>

**B.** In a long-term memory store, retrieved or loaded into the context at session start

Long-term memory is storage outside the window that is retrieved when relevant, like RAG over your own notes.

- **A:** History ends with the session.
- **B:** Correct.
- **C:** Fine-tuning per user preference is costly, slow, and hard to undo.
- **D:** The reserve is space for the answer, not storage.

</details>

## 9. Multiple choice (hard)

Users ask repeated questions over a 2,000-page knowledge base, and your model has a one-million-token window. What is the best default design?

- **A.** Put the whole knowledge base in every call
- **B.** Retrieve a few relevant chunks per question, put the stable instructions first, and measure answer quality on a labelled set
- **C.** Fine-tune the model on the knowledge base and send no documents
- **D.** Summarize the whole knowledge base into one page and send that

<details>
<summary>Answer</summary>

**B.** Retrieve a few relevant chunks per question, put the stable instructions first, and measure answer quality on a labelled set

Retrieval keeps cost and latency low and avoids burying the relevant passage among irrelevant text.

- **A:** Huge per-call cost and prefill latency, and accuracy can drop with irrelevant text.
- **B:** Correct.
- **C:** Fine-tuning is poor at storing changing facts and gives no citations.
- **D:** A one-page summary loses the details most questions need.

</details>
