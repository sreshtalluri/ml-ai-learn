import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-4 pt-24 text-center">
      <h1 className="text-3xl font-semibold tracking-tight">Page not found</h1>
      <p className="mt-3 text-muted">That page does not exist. The learning path lists every lesson.</p>
      <Link href="/path/" className="mt-6 inline-block rounded-lg bg-accent px-4 py-2 font-medium text-white dark:text-zinc-950">Go to the learning path</Link>
    </div>
  );
}
