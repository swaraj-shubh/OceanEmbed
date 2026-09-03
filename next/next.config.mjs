// Proxy API calls to the FastAPI backend so the browser sees one origin (no CORS).
// Override the target with API_PROXY_TARGET (e.g. a deployed backend URL).
const API = process.env.API_PROXY_TARGET || "http://localhost:8000";

/** @type {import('next').NextConfig} */
export default {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API}/api/:path*` },
      { source: "/readyz", destination: `${API}/readyz` },
      { source: "/healthz", destination: `${API}/healthz` },
    ];
  },
};
