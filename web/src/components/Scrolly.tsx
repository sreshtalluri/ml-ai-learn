"use client";
// Scrolling explainer: text on one side, the lab pinned on the other. The section crossing the
// middle of the viewport becomes active, and its step is sent to the lab through TourContext.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { LabEmbed } from "./labs/LabEmbed";
import { TourContext } from "./labs/tour";

export function Scrolly({ lab, sections }: { lab: string; sections: { step: string | null; content: ReactNode }[] }) {
  const refs = useRef<(HTMLElement | null)[]>([]);
  const [active, setActive] = useState(0);

  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(Number((e.target as HTMLElement).dataset.index));
      },
      // a thin band across the middle of the screen; on phones, the middle of the text area below the pinned lab
      { rootMargin: window.matchMedia("(min-width: 1024px)").matches ? "-45% 0px -50% 0px" : "-70% 0px -25% 0px" },
    );
    refs.current.forEach((el) => el && io.observe(el));
    return () => io.disconnect();
  }, [sections.length]);

  // the most recent step at or above the active section (prose-only sections keep the lab where it is)
  const step = sections.slice(0, active + 1).findLast((s) => s.step)?.step ?? null;

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,28rem)_minmax(0,1fr)] lg:gap-10">
      <div className="sticky top-16 z-10 -mx-4 max-h-[46vh] overflow-y-auto border-b border-line bg-bg px-4 lg:order-2 lg:top-20 lg:mx-0 lg:max-h-[calc(100dvh-6rem)] lg:border-0 lg:px-0 lg:self-start">
        <TourContext.Provider value={{ step }}>
          <div className="[&>section]:my-0 [&>section]:lg:my-0"><LabEmbed id={lab} /></div>
        </TourContext.Provider>
      </div>
      <div className="lg:order-1 pb-[40vh]">
        {sections.map((s, i) => (
          <section
            key={i}
            ref={(el) => { refs.current[i] = el; }}
            data-index={i}
            className={`prose py-6 transition-opacity duration-300 lg:min-h-[55vh] ${i === active ? "opacity-100" : "opacity-45"}`}
          >
            {s.content}
          </section>
        ))}
      </div>
    </div>
  );
}
