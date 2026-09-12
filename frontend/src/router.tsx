import { createRootRoute, createRoute } from "@tanstack/react-router";
import { AdaptersPage } from "./routes/adapters";
import { EventsPage } from "./routes/events";
import { FleetPage } from "./routes/fleet";
import { JobsPage } from "./routes/jobs";
import { OverviewPage } from "./routes/overview";
import { RolloutsPage } from "./routes/rollouts";
import { RunDetailPage } from "./routes/run-detail";
import { TrainingPage } from "./routes/training";
import { TrainingRunDetailPage } from "./routes/training-detail";
import { App } from "./App";

const rootRoute = createRootRoute({ component: App });

const routes = [
  createRoute({ getParentRoute: () => rootRoute, path: "/", component: OverviewPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/rollouts", component: RolloutsPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/rollouts/$runId", component: RunDetailPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/training", component: TrainingPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/training/$trainingRunId", component: TrainingRunDetailPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/adapters", component: AdaptersPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/fleet", component: FleetPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/jobs", component: JobsPage }),
  createRoute({ getParentRoute: () => rootRoute, path: "/events", component: EventsPage }),
];

export const routeTree = rootRoute.addChildren(routes);
