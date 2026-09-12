import { Badge, Button, Flex, Text } from "@radix-ui/themes";
import { Link, Outlet } from "@tanstack/react-router";
import { Activity, Boxes, Cpu, GitBranch, Layers, ListChecks, Radio, ScrollText } from "lucide-react";
import { useBudget, useLiveInvalidation, useRuntimeConfig } from "../lib/query";
import { useSocketStatus } from "../lib/hub";
import { fmtUsd } from "../lib/format";

const NAV = [
  { to: "/", label: "Overview", icon: Activity, exact: true },
  { to: "/rollouts", label: "Rollouts", icon: Layers },
  { to: "/training", label: "Training", icon: GitBranch },
  { to: "/adapters", label: "Adapters", icon: Boxes },
  { to: "/fleet", label: "Fleet", icon: Cpu },
  { to: "/jobs", label: "Jobs", icon: ListChecks },
  { to: "/events", label: "Events", icon: ScrollText },
] as const;

export function Shell({ appearance, onToggleTheme }: { appearance: "dark" | "light"; onToggleTheme: () => void }) {
  useLiveInvalidation();
  const socket = useSocketStatus();
  const config = useRuntimeConfig();
  const budget = useBudget();
  return (
    <div className="shell">
      <aside className="rail">
        <div className="rail-brand">
          <Text size="1" className="muted" style={{ letterSpacing: "0.06em", textTransform: "uppercase" }}>Open Horizon</Text>
          <br />
          <Text size="3" weight="bold">RL control plane</Text>
        </div>
        <nav className="rail-nav">
          {NAV.map(({ to, label, icon: Icon, ...rest }) => (
            <Link key={to} to={to} activeOptions={{ exact: "exact" in rest && rest.exact }}>
              <Icon /> {label}
            </Link>
          ))}
        </nav>
        <div className="rail-foot">
          <Text size="1" className="muted">
            {config.data ? `${config.data.env_backend} · ${config.data.trainer_backend} · ${config.data.store_backend}` : ""}
          </Text>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <Flex align="center" gap="2">
            <Radio size={13} color={socket === "open" ? "var(--status-good)" : "var(--status-critical)"} />
            <Text size="1" className="muted">events {socket}</Text>
          </Flex>
          {config.data ? (
            <Badge variant="outline" color="gray" radius="small">
              <span className="mono">{config.data.policy_name}</span>
            </Badge>
          ) : null}
          {config.data ? <Text size="1" className="muted">worker {config.data.worker_id}</Text> : null}
          {budget.data && budget.data.cap_usd > 0 ? (
            <Badge variant="soft" color={budget.data.exceeded ? "red" : "gray"} radius="small">
              budget {fmtUsd(budget.data.spent_usd)} / {fmtUsd(budget.data.cap_usd, 2)}
            </Badge>
          ) : null}
          <div style={{ marginLeft: "auto" }}>
            <Button size="1" variant="ghost" color="gray" onClick={onToggleTheme}>{appearance === "dark" ? "light" : "dark"}</Button>
          </div>
        </header>
        <main className="page">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
