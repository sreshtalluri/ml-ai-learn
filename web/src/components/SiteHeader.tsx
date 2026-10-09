"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { List, Moon, Sun, X } from "@phosphor-icons/react";

export const NAV = [
  { href: "/path/", label: "Learn" },
  { href: "/labs/", label: "Labs" },
  { href: "/models/", label: "Models" },
  { href: "/math/", label: "Math Lab" },
  { href: "/sprints/", label: "Interview prep" },
  { href: "/cheatsheets/", label: "Cheat sheets" },
  { href: "/glossary/", label: "Glossary" },
  { href: "/progress/", label: "Progress" },
];

function ThemeToggle() {
  const toggle = () => {
    const dark = document.documentElement.classList.toggle("dark");
    localStorage.setItem("theme", dark ? "dark" : "light");
  };
  return (
    <button onClick={toggle} aria-label="Toggle dark mode" className="rounded-lg p-2 text-muted hover:text-ink hover:bg-surface-2">
      <Sun className="hidden dark:block" size={18} />
      <Moon className="block dark:hidden" size={18} />
    </button>
  );
}

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const active = (href: string) => pathname.startsWith(href) || (href === "/path/" && (pathname.startsWith("/learn") || pathname.startsWith("/modules"))) || (href === "/sprints/" && (pathname.startsWith("/drill") || pathname.startsWith("/quizzes")));

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-6 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight shrink-0" onClick={() => setOpen(false)}>
          <span aria-hidden className="grid h-7 w-7 place-items-center rounded-lg bg-ink text-bg text-[13px] font-bold">ŷ</span>
          <span>ML to LLMs</span>
        </Link>
        <nav aria-label="Main" className="hidden lg:flex items-center gap-1 text-sm">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              aria-current={active(n.href) ? "page" : undefined}
              className={`rounded-lg px-2.5 py-1.5 whitespace-nowrap ${active(n.href) ? "text-ink bg-surface-2 font-medium" : "text-muted hover:text-ink"}`}
            >
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1">
          <ThemeToggle />
          <button className="lg:hidden rounded-lg p-2 text-muted hover:text-ink" aria-label="Menu" aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? <X size={20} /> : <List size={20} />}
          </button>
        </div>
      </div>
      {open && (
        <nav aria-label="Main" className="lg:hidden border-t border-line bg-bg px-4 py-3 grid grid-cols-2 gap-1">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href} onClick={() => setOpen(false)} className={`rounded-lg px-3 py-2 ${active(n.href) ? "bg-surface-2 text-ink" : "text-muted"}`}>
              {n.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
