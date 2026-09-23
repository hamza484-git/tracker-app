import { useEffect } from "react";

type Theme = "dark" | "light";

function getThemeFromUrl(): Theme | null {
  const param = new URLSearchParams(window.location.search).get("theme");
  return param === "light" || param === "dark" ? param : null;
}

function getFallbackTheme(): Theme {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function ThemeToggle() {
  useEffect(() => {
    const initial = getThemeFromUrl() ?? getFallbackTheme();
    document.documentElement.dataset.theme = initial;

    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === "crickethub:theme" && (event.data.theme === "light" || event.data.theme === "dark")) {
        document.documentElement.dataset.theme = event.data.theme;
      }
    };
    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  return null;
}