import { Outlet, createRootRoute, createRoute } from "@tanstack/react-router";
import { DashboardPage } from "./routes/dashboard";
import { RunDetailPage } from "./routes/run-detail";

const rootRoute = createRootRoute({
  component: () => (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Observability</p>
          <h1>RL Stack Dashboard</h1>
        </div>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  ),
});

const dashboardRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: DashboardPage,
});

const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  component: RunDetailPage,
});

export const routeTree = rootRoute.addChildren([dashboardRoute, runDetailRoute]);
