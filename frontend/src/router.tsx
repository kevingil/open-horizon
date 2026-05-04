import { Link, Outlet, createRootRoute, createRoute } from "@tanstack/react-router";
import { AdapterListPage } from "./routes/adapters";
import { DashboardPage } from "./routes/dashboard";
import { RunDetailPage } from "./routes/run-detail";
import { TrainingListPage } from "./routes/training";
import { TrainingRunDetailPage } from "./routes/training-detail";

const rootRoute = createRootRoute({
  component: () => (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Observability</p>
          <h1>Distributed RL Dashboard</h1>
        </div>
        <nav className="app-nav">
          <Link to="/" activeOptions={{ exact: true }}>
            Rollouts
          </Link>
          <Link to="/training">Training</Link>
          <Link to="/adapters">Adapters</Link>
        </nav>
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

const trainingListRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/training",
  component: TrainingListPage,
});

const trainingDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/training/$trainingRunId",
  component: TrainingRunDetailPage,
});

const adaptersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/adapters",
  component: AdapterListPage,
});

export const routeTree = rootRoute.addChildren([
  dashboardRoute,
  runDetailRoute,
  trainingListRoute,
  trainingDetailRoute,
  adaptersRoute,
]);
