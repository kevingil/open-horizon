/**
 * Event stream types. One envelope per event:
 *   { seq, at, subject?: {kind, id}, kind, payload }
 * `kind` discriminates `payload`; `subject` says what the event is about.
 */
import type { components } from "./api-schema";

type S = components["schemas"];

export type DomainEvent = S["DomainEvent"];
export type Subject = S["Subject"];
export type EventEnvelope = S["EventEnvelope"];

export type EventKind = DomainEvent["kind"];
export type EventOf<K extends EventKind> = Extract<EventEnvelope, { kind: K }>;

export function subjectId(event: EventEnvelope, kind: Subject["kind"]): string | null {
  const s = event.subject;
  return s && s.kind === kind ? s.id : null;
}

export function runIdOf(event: EventEnvelope): string | null {
  return subjectId(event, "run");
}
