// Server component: renders guide markdown with the same semantics GitHub uses,
// plus site-only upgrades (labs, quizzes, glossary hovers, route-aware links).
import path from "node:path";
import Link from "next/link";
import type { ReactElement, ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { getGlossary, getQuiz, REPO_URL, routeForFile, slugify } from "@/lib/content";
import { BASE_PATH, SITE_URL } from "@/lib/site";
import { CodeBlock } from "./CodeBlock";
import { GlossaryLink } from "./GlossaryLink";
import { LabEmbed } from "./labs/LabEmbed";
import { QuizRunner } from "./quiz/QuizRunner";

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Replace invisible GitHub markers with fenced blocks the renderer can intercept. */
export function preprocess(src: string): string {
  return src
    .replace(/<!--\s*lab:([\w-]+)\s*-->[\s\S]*?<!--\s*\/lab\s*-->/g, "\n```lab\n$1\n```\n")
    .replace(/<!--\s*quiz:([\w-]+)\s*-->[\s\S]*?<!--\s*\/quiz\s*-->/g, "\n```quiz\n$1\n```\n");
}

const ALERT = /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/;

/** remark plugin: GitHub alerts (> [!NOTE]) and the lesson "Mental model" quote become styled callouts. */
function remarkCallouts() {
  return (tree: any) => {
    const walk = (node: any) => {
      if (node.type === "blockquote") {
        const para = node.children?.[0];
        const first = para?.children?.[0];
        if (first?.type === "text" && ALERT.test(first.value)) {
          const kind = first.value.match(ALERT)[1].toLowerCase();
          first.value = first.value.replace(ALERT, "");
          if (!first.value) para.children.shift();
          if (para.children[0]?.type === "break") para.children.shift();
          node.data = { hProperties: { className: ["callout", `callout-${kind}`], "data-title": kind } };
        } else if (first?.type === "strong" && /^Mental model/i.test(first.children?.[0]?.value ?? "")) {
          node.data = { hProperties: { className: ["callout", "callout-mental"] } };
        }
      }
      node.children?.forEach(walk);
    };
    walk(tree);
  };
}

const TITLES: Record<string, string> = { note: "Note", tip: "Engineering note", important: "Key distinction", warning: "Watch out", caution: "Caution" };

function resolveHref(href: string, file: string): { kind: "internal" | "external" | "glossary"; href: string; term?: string } {
  if (href.startsWith(SITE_URL)) return { kind: "internal", href: href.slice(SITE_URL.length - 1) || "/" };
  if (/^[a-z]+:/i.test(href) || href.startsWith("//")) return { kind: "external", href };
  const [p, hash] = href.split("#");
  if (!p) return { kind: "internal", href: `#${hash}` };
  const rel = path.posix.normalize(path.posix.join(path.posix.dirname(file), p));
  const route = routeForFile(rel);
  if (route === "/glossary/" && hash) return { kind: "glossary", href: `${route}#${hash}`, term: hash };
  if (route) return { kind: "internal", href: hash ? `${route}#${hash}` : route };
  return { kind: "external", href: `${REPO_URL}/blob/main/guide/${rel}` };
}

function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (node && typeof node === "object" && "props" in node) return textOf((node as ReactElement<{ children?: ReactNode }>).props.children);
  return "";
}

export function Markdown({ source, file }: { source: string; file: string }) {
  const glossary = new Map(getGlossary().map((g) => [g.id, g]));

  const components: Components = {
    a({ href = "", children }) {
      const r = resolveHref(href, file);
      if (r.kind === "glossary") {
        const entry = glossary.get(r.term!) ?? glossary.get(slugify(r.term!));
        return <GlossaryLink href={r.href} definition={entry?.fields["Plain English"]} term={entry?.term}>{children}</GlossaryLink>;
      }
      if (r.kind === "internal") return <Link href={r.href}>{children}</Link>;
      return <a href={r.href} target="_blank" rel="noreferrer">{children}</a>;
    },
    img({ src = "", alt }) {
      const s = String(src);
      const url = /^https?:/.test(s) ? s : `${BASE_PATH}/guide-assets/${path.posix.normalize(path.posix.join(path.posix.dirname(file), s))}`;
      // eslint-disable-next-line @next/next/no-img-element
      return <img src={url} alt={alt ?? ""} loading="lazy" />;
    },
    pre({ children }) {
      const code = children as ReactElement<{ className?: string; children?: ReactNode }>;
      const lang = code?.props?.className?.match(/language-([\w-]+)/)?.[1];
      const text = textOf(code?.props?.children).replace(/\n$/, "");
      if (lang === "lab") return <LabEmbed id={text.trim()} />;
      if (lang === "quiz") {
        const quiz = getQuiz(text.trim());
        return quiz ? <QuizRunner quiz={quiz} /> : null;
      }
      return <CodeBlock code={text} lang={lang} />;
    },
    table({ children }) {
      return <div className="overflow-x-auto rounded-xl border border-line bg-surface"><table>{children}</table></div>;
    },
    blockquote({ node, children, ...props }: any) {
      const title = props["data-title"];
      return (
        <blockquote className={props.className}>
          {title && <span className="callout-title">{TITLES[title] ?? title}</span>}
          {children}
        </blockquote>
      );
    },
  };

  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkCallouts]}
        rehypePlugins={[rehypeRaw, rehypeSlug, [rehypeKatex, { throwOnError: false, strict: false }]]}
        components={components}
      >
        {preprocess(source)}
      </ReactMarkdown>
    </div>
  );
}
