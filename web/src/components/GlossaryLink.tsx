"use client";
import Link from "next/link";
import { useId, useState, type ReactNode } from "react";

/** A glossary link that shows the plain-English definition on hover or keyboard focus. */
export function GlossaryLink({ href, term, definition, children }: { href: string; term?: string; definition?: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  if (!definition) return <Link href={href}>{children}</Link>;
  return (
    <span className="relative inline-block" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <Link
        href={href}
        aria-describedby={open ? id : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="decoration-dotted"
      >
        {children}
      </Link>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute left-0 top-full z-20 mt-1 w-72 rounded-xl border border-line bg-surface p-3 text-sm leading-snug text-ink shadow-lg not-italic font-normal"
        >
          <span className="block font-semibold mb-1">{term}</span>
          {definition.replace(/\$([^$]+)\$/g, "$1")}
        </span>
      )}
    </span>
  );
}
