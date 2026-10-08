import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SiteHeader } from "@/components/SiteHeader";
import { REPO_URL } from "@/lib/content";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "ML to LLMs: an interactive course", template: "%s · ML to LLMs" },
  description:
    "A visual, math-first course from linear regression to transformers, RAG and production AI engineering. Interactive labs, worked examples, quizzes, and local progress tracking.",
};

// Runs before paint so the saved theme never flashes.
const themeScript = `try{var t=localStorage.getItem("theme");if(t==="dark"||(!t&&matchMedia("(prefers-color-scheme: dark)").matches))document.documentElement.classList.add("dark")}catch(e){}`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-[100dvh] flex flex-col font-sans">
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2">
          Skip to content
        </a>
        <SiteHeader />
        <main id="main" className="flex-1">{children}</main>
        <footer className="border-t border-line mt-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 py-8 text-sm text-faint flex flex-wrap gap-x-6 gap-y-2 justify-between">
            <p>Every lesson is also readable as plain markdown on GitHub. Progress is stored only in this browser.</p>
            <a className="hover:text-ink underline underline-offset-4" href={REPO_URL}>Source and markdown course on GitHub</a>
          </div>
        </footer>
      </body>
    </html>
  );
}
