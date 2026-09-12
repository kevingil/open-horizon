import { Card, Heading } from "@radix-ui/themes";
import type { ReactNode } from "react";

export function Panel({ title, right, flush, children }: { title: ReactNode; right?: ReactNode; flush?: boolean; children: ReactNode }) {
  return (
    <Card className="panel" variant="surface">
      <div className="panel-head">
        <Heading size="2" weight="medium">{title}</Heading>
        <div className="muted" style={{ fontSize: 12 }}>{right}</div>
      </div>
      <div className={`panel-body${flush ? " flush" : ""}`}>{children}</div>
    </Card>
  );
}
