export const dynamic = "force-dynamic";

import { NextResponse } from "next/server";

export async function GET() {
  const baseUrl =
    process.env.R2_PUBLIC_URL ||
    `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`;
  const pdfUrl = `${baseUrl}/${process.env.R2_BUCKET_NAME}/resume.pdf`;

  const response = await fetch(pdfUrl);
  if (!response.ok || !response.body) {
    return new NextResponse("Not Found", { status: 404 });
  }

  return new Response(await response.arrayBuffer(), {
    status: 200,
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": "inline; filename=\"resume.pdf\"",
    },
  });
}
