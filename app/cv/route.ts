export const dynamic = "force-static";

import fs from "fs";
import path from "path";

export async function GET() {
  const filePath = path.join(process.cwd(), "public", "resume.pdf");
  const file = fs.readFileSync(filePath);
  return new Response(file, {
    status: 200,
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": "inline; filename=\"resume.pdf\"",
    },
  });
}
