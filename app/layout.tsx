import type { Metadata } from "next";
import Script from "next/script";
import { DM_Sans, JetBrains_Mono } from "next/font/google";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { siteConfig } from "@/lib/site-config";
import { Analytics } from "@vercel/analytics/next"
import "./globals.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
  weight: ["400", "500", "600"],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: siteConfig.title,
    template: `%s | ${siteConfig.title}`,
  },
  description: siteConfig.description,
  metadataBase: new URL(siteConfig.baseUrl),
  alternates: {
    types: {
      "application/rss+xml": "/rss.xml",
    },
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="light">
      <head>
        <Script
          id="theme-init"
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                const themeConfig = ${JSON.stringify(siteConfig.theme)};
                if (!themeConfig.toggleEnabled) {
                  document.documentElement.setAttribute("data-theme", themeConfig.defaultTheme);
                } else {
                  const stored = localStorage.getItem("theme");
                  if (stored) {
                    document.documentElement.setAttribute("data-theme", stored);
                  } else {
                    document.documentElement.setAttribute("data-theme", themeConfig.defaultTheme);
                  }
                }
              })();
            `,
          }}
        />
      </head>
      <body className={`${dmSans.variable} ${jetbrainsMono.variable}`}>
        <Header />
        <Analytics />
        <main className="main-content">
          <div className="container">{children}</div>
        </main>
        <Footer />
      </body>
    </html>
  );
}
