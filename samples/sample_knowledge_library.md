# Knowledge Library — Shreyas Kumar

Extra context the resume never had room for. (Note: this file adds richer
metrics for facts already in the resume, plus new facts — including Grafana/
Prometheus experience the JD lists as nice-to-have. It deliberately still has
NO Kubernetes experience, so the Do-Not-Claim list must keep it excluded.)

## Finlytics — war stories

- The Redis caching project also involved designing cache-invalidation logic
  for 14 event types; wrote an internal RFC that was adopted platform-wide.
- On-call for the payments analytics platform (PagerDuty rotation, 1 week in
  4); drove MTTR from 45 min to 12 min by building Grafana dashboards backed
  by Prometheus metrics across all 12 microservices.
- Presented the event-sourcing schema design at Austin Python Meetup (2024),
  ~80 attendees.
- Worked directly with the compliance team preparing evidence for the
  company's SOC 2 Type II audit (access controls, audit logging).

## DataHarbor — extra detail

- The Kafka ingestion service used exactly-once semantics via idempotent
  producers; peak sustained load was actually 22k messages/sec during
  quarter-end (15k was the steady-state number on the resume).
- Ran fortnightly knowledge-sharing sessions on pytest patterns for a team
  of 6.

## Writing & community

- Blog post "Async SQLAlchemy in production" — 30k views on dev.to.
- Maintainer duties on JobTrackr: triaged 150+ issues, reviewed 60+ external
  PRs, wrote the contributor guide.
