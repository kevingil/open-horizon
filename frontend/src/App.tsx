import { Theme } from "@radix-ui/themes";
import { useEffect, useState } from "react";
import { Shell } from "./components/Shell";

function initialTheme(): "dark" | "light" {
  try {
    const saved = localStorage.getItem("horizon.theme");
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    // ignore
  }
  return "dark";
}

export function App() {
  const [appearance, setAppearance] = useState<"dark" | "light">(initialTheme);
  useEffect(() => {
    try {
      localStorage.setItem("horizon.theme", appearance);
    } catch {
      // ignore
    }
    document.documentElement.classList.toggle("dark", appearance === "dark");
  }, [appearance]);
  return (
    <Theme appearance={appearance} accentColor="blue" grayColor="slate" radius="small" scaling="95%" panelBackground="solid">
      <Shell appearance={appearance} onToggleTheme={() => setAppearance((a) => (a === "dark" ? "light" : "dark"))} />
    </Theme>
  );
}
