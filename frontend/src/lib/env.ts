import { z } from "zod";

const schema = z.object({
  NEXT_PUBLIC_API_URL: z.string().url().default("http://localhost:8000"),
});

const parsed = schema.safeParse({
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
});

if (!parsed.success) {
  console.warn("Invalid env, falling back to defaults:", parsed.error.flatten());
}

export const env = parsed.success ? parsed.data : { NEXT_PUBLIC_API_URL: "http://localhost:8000" };

export const API_URL = env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
