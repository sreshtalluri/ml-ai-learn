import type { NextConfig } from "next";

// Static export so the site runs anywhere (GitHub Pages sets NEXT_PUBLIC_BASE_PATH=/ml-ai-learn).
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
