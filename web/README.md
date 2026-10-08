# ML to LLMs: website

The interactive edition of the course. It reads all content from [`../guide`](../guide/) at build time; see the [root README](../README.md) for the architecture and [`../guide/AUTHORING.md`](../guide/AUTHORING.md) for the content format.

```bash
npm ci
npm run dev            # http://localhost:3000 (copies guide figures into public/guide-assets first)
npm test               # unit, component, and content-integrity tests
npm run typecheck      # run `npx next typegen` once first on a fresh clone
npm run lint
npm run gen:quizzes    # regenerate guide/quizzes/*.md after editing a quiz .yml
npm run build          # static export to ./out
```

Set `NEXT_PUBLIC_BASE_PATH=/ml-ai-learn` when building for GitHub Pages.
