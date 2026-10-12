---
title: How retrieval-augmented generation finds its evidence
summary: Follow one question through a RAG pipeline (chunking, lexical scoring, a vocabulary-mismatch miss, semantic matching, top-k and reranking) and see why no prompt can rescue evidence that was never retrieved.
lab: rag
lesson: rag-pipeline
minutes: 7
---

# How retrieval-augmented generation finds its evidence

<!-- lab:rag -->
![Left: evidence recall at 3 rises from 0.4 with 8-word chunks to 1.0 with 50-word chunks, while the number of context words sent to the model rises linearly. Right: a bar chart of five ranked results with relevant results at ranks 2 and 4, annotated with discounted gains, giving recall@3 0.5, reciprocal rank 0.5, and nDCG@5 0.651.](../figures/rag-pipeline.png)
<!-- /lab -->

A RAG system answers from your documents. The model only sees what retrieval hands it, though, so the whole pipeline comes down to one question: does the sentence with the answer reach the prompt? This explainer follows that sentence through four small synthetic policy documents. On the website, the lab beside the text changes as you scroll.

<!-- step:chunks -->
## Cutting documents into chunks

Retrieval doesn't fetch whole documents. It fetches **chunks**: windows of $S$ words that start every $S - O$ words, where $O$ is the overlap. A document of $N$ words gives about

```math
\left\lceil \frac{N - O}{S - O} \right\rceil
```

chunks. With 60-word chunks our four documents are 4 chunks, one each. Shrink to 20 words with 5 words of overlap and the index grows to 10.

The size involves a trade-off. Small chunks are precise but can cut the answer sentence in half. Large ones keep it whole, but they bury it in unrelated text and cost more tokens later. Overlap makes it less likely that a key sentence lands across a boundary.

<!-- step:question -->
## The question becomes query terms

Our first question is *"Are shipping costs refunded if my item was broken on arrival?"* The answer lives in the refunds document: "Shipping fees are not refundable unless the item **arrived damaged**."

A lexical retriever doesn't read the question the way you do. It lowercases it, drops common words like *are*, *if* and *my*, and keeps the rest: *shipping, costs, refunded, item, broken, arrival*. Then it scores those terms against every chunk. Notice already that *refunded* is not *refundable* and *broken* is not *damaged*. Those near-misses will matter in a moment.

<!-- step:lexical -->
## Lexical retrieval and its ranking

Each chunk and the query become TF-IDF vectors. A word counts more when it is rare across chunks (weight $\ln N/\mathrm{df}$), and chunks are ranked by cosine similarity to the query.

In the lab, the refunds chunk with the evidence ranks first (cosine 0.408), but a shipping chunk sits close behind (0.347) only because it also says *shipping*. The ranking is fragile. Change the chunk size and the two swap places: at 30 words the shipping chunk wins. Lexical retrieval matches words, not meaning, so a strong match on one incidental word can beat the chunk that actually answers.

<!-- step:miss -->
## A miss from vocabulary mismatch

Now ask *"Can I get reimbursed for a game I never opened?"* The answer is in the collection: "Digital downloads are refundable only if they have **not been accessed**."

But look at the words. *Reimbursed* versus *refundable*, *game* versus *digital downloads*, *never opened* versus *not accessed*. After stop words are removed, the question and the documents share no term at all, so every TF-IDF score is exactly 0 and nothing is retrieved. Nothing crashed and the index is fine. The user simply phrased the question differently from the people who wrote the policy, and that happens all the time in real systems.

<!-- step:semantic -->
## Semantic matching fixes it

Dense retrievers embed text so that phrases with similar meanings land near each other. The lab simulates this with a small synonym map: *reimbursed* also matches *refund*, *game* matches *digital downloads*, *opened* matches *accessed*.

Turn it on and the same question puts the refunds chunk with the evidence at **rank 1**. That's why production systems often run **hybrid** retrieval, combining both scores:

```math
s = \lambda \, s_{\text{dense}} + (1 - \lambda) \, s_{\text{lexical}}
```

Lexical matching nails exact identifiers like error codes and product names, and dense matching catches paraphrases.

<!-- step:topk -->
## Top-k and the context budget

Only the top $k$ chunks are pasted into the prompt. Ask *"How many days do I have to ask for my money back?"* with lexical retrieval and two shipping chunks (which also mention *days*) plus a different refunds chunk all outrank the refund-window chunk. The evidence is at **rank 4**.

Sweep $k$ from 1 to 4. The evidence enters the context exactly at $k = 4$. Meanwhile the "context sent" counter climbs with every chunk, and each extra chunk adds tokens, latency, and text the model can be distracted by. Raising $k$ until the answer shows up works, but you pay for it on every request.

<!-- step:rerank -->
## Reranking

A cheaper fix is to retrieve generously and then sort more carefully. A **reranker**, usually a cross-encoder, reads the query and each candidate together and rescores them. It is too slow to run on the whole index, so it runs on the top 10 or so.

The lab's simulated reranker rewards chunks where the query terms (and their synonyms) appear close together. With $k = 1$ and reranking on, the refund chunk jumps from rank 4 to **rank 1**. You get the right evidence and send one chunk instead of four.

<!-- step:no-prompt-fix -->
## No prompt fixes a retrieval miss

Go back to the reimbursement question with lexical retrieval. The evidence is not in the context. A well-instructed model says "I don't know"; a poorly instructed one invents a refund policy. Rewording the system prompt can only change which of those two you get. It can't produce a fact the model never saw.

So when a RAG answer is wrong, check retrieval first: was the evidence retrieved, and at what rank? Measure that separately from answer quality, and fix it with chunking, hybrid retrieval, $k$, or reranking before touching the prompt.

**Try it yourself:** type your own question into the lab and judge whether the top chunks could answer it, or open the [full lesson](../lessons/16-rag/01-rag-pipeline.md) for the math and a runnable pipeline.
