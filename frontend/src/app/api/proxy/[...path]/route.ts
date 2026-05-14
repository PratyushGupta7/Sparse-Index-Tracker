import { API_URL } from "@/lib/env";
import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

async function proxy(req: NextRequest, path: string[]): Promise<Response> {
  const search = req.nextUrl.search;
  const target = `${API_URL}/${path.join("/")}${search}`;
  const init: RequestInit = {
    method: req.method,
    headers: { "Content-Type": "application/json" },
    body: req.method === "GET" || req.method === "HEAD" ? undefined : await req.text(),
    signal: req.signal,
  };
  try {
    const upstream = await fetch(target, init);
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
      },
    });
  } catch (err) {
    return NextResponse.json({ detail: `Proxy fetch failed: ${String(err)}` }, { status: 502 });
  }
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  return proxy(req, path);
}
