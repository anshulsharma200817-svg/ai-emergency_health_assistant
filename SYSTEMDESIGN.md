# SYSTEMDESIGN.md

**Project:** Hackathon submission & judging platform (Dogfood 2026 prep)
**Stack:** Django 6.1 + Django REST Framework + PostgreSQL 16 + React (Vite, TypeScript)
**Deployment target:** one machine, `docker compose up`, fully offline

---

## 0. The governing constraint

Before any design decision, this problem statement fixes the operating envelope:

| Constraint | Consequence for design |
|---|---|
| Must run from `docker compose up` on a laptop | Single node. No cluster, no service mesh, no multi-region. |
| Must work with **network disabled** | No CDN, no hosted cache, no external auth, no third-party API, no telemetry SaaS. |
| Must be self-hosted by a small organizer team | Every moving part is an operational burden. Fewer containers = higher score. |
| Realistic load: ~300 projects, ~50 judges, a few thousand voters | This is a **small-data** problem. Postgres on a laptop handles it comfortably. |
| Scored on correctness (40%), judging integrity (25%), operability (20%), code quality (15%) | Correctness and operability beat throughput engineering. |

> **The single most important design judgment in this project is knowing which "real-world system design" techniques to leave out.** A Kafka queue in a laptop-deployed hackathon platform is not sophistication; it is a bug in judgment. This document records both sides of that decision explicitly.

---

## 1. Scope decision table — what we implement, what we skip

### 1.1 Implement (earns correctness, integrity or operability points)

| # | Technique | Why it earns its place here | Cost |
|---|---|---|---|
| 1 | **Database connection pooling** (`CONN_MAX_AGE`, `CONN_HEALTH_CHECKS`) | Django opens a new Postgres connection per request by default. Under the acceptance suite's concurrent calls this is the first thing that hurts. One settings line. | 2 lines |
| 2 | **Deliberate indexing** (FK indexes, composite indexes on hot filters, GIN index for gallery search) | Every list endpoint filters by `event_id` + status. Without indexes, sequential scans; with them, the gallery stays instant. Also the clearest "we thought about the database" signal in code review. | Migration |
| 3 | **N+1 query elimination** (`select_related` / `prefetch_related`) | The judge dashboard and gallery are join-heavy. This is the single most common performance defect in Django code and reviewers look for it. | Discipline |
| 4 | **Keyset-friendly pagination on every list endpoint** | Prevents unbounded responses, required for a gallery of 300+ projects, and is table stakes for API quality. | DRF config |
| 5 | **Application-level caching with explicit invalidation** (see §4) | Leaderboards and normalization runs are expensive to recompute per request. Caching them is defensible. Caching them *without* documented invalidation is worse than not caching. | Moderate |
| 6 | **Rate limiting** (DRF throttling, DB/locmem-backed) | Directly scored: T3 requires rate limiting, and the threat model bonus depends on it. Applies to register, login, vote, comment. | Low |
| 7 | **Idempotency on mutating endpoints** (unique constraints + upsert semantics) | Double-clicked "submit vote" or a retried webhook must not create two rows. Enforced in the DB, not in JavaScript. | Constraints |
| 8 | **Transactions + row locking for scoring** (`transaction.atomic`, `select_for_update`) | Two judges scoring the same submission concurrently, or a normalization run reading mid-write, must not corrupt results. This is judging integrity — 25% of the score. | Moderate |
| 9 | **Append-only audit log** | Required by T3, and it is what makes every other judging claim defensible. Write on every state transition. | Moderate |
| 10 | **Background job execution for slow work** (see §5 — a simple in-process/threaded worker, not a broker) | Normalization runs, CSV exports, certificate generation and webhook delivery must not block an HTTP request. | Moderate |
| 11 | **Secure-by-default authorization** (DRF `DEFAULT_PERMISSION_CLASSES = IsAuthenticated`, opt-in public routes) | The failure mode that loses this hackathon is a judge endpoint that a participant can call. Default-deny makes that structurally hard. | 1 line |
| 12 | **Structured logging + `/healthz` endpoint** | Operability. Lets an organizer see what happened and lets Docker health-check the app. | Low |
| 13 | **Migrations as the only schema path** | Reproducibility on a fresh clone. Never `--fake`, never hand-edited SQL. | Discipline |
| 14 | **Seed/fixture loading that is idempotent** | `docker compose up` twice must not duplicate the fixture data. This is explicitly what the graders will do. | Low |

### 1.2 Deliberately NOT implemented (and why)

| Technique | Why it is wrong *for this problem* |
|---|---|
| **Horizontal scaling / load balancer (nginx round-robin over N app nodes)** | The deliverable is a laptop deployment. A load balancer across one replica is theatre. *We do run a reverse proxy (§3), but for static-file serving and a single entrypoint, not for balancing.* |
| **Redis / Memcached as a separate container** | Adds a container, a failure mode and a dependency to a system whose entire working set fits in Postgres and process memory. Postgres-backed cache + per-process locmem is enough at this scale. Revisit only if profiling shows otherwise. |
| **Celery + broker (RabbitMQ/Redis)** | Three extra containers to run jobs that take under a second at this data size. §5 explains the lighter alternative. |
| **Kafka / event streaming** | There is no stream. There are a few thousand rows. |
| **Microservices** | Splitting a 72-hour single-team codebase into services multiplies deployment and debugging cost while removing transactional integrity — exactly the property the judging engine needs most. Modular monolith instead (§2). |
| **CDN / edge caching** | Offline requirement makes this impossible by definition. |
| **Read replicas / sharding / partitioning** | Wrong by three orders of magnitude. |
| **Kubernetes / Helm** | The acceptance criterion is literally `docker compose up`. |
| **Elasticsearch / OpenSearch** | Postgres full-text search (`tsvector` + GIN) covers gallery search with zero extra infrastructure. |
| **S3 / object storage** | A Docker volume on local disk. Offline requirement again. |
| **External auth (OAuth, Auth0, Clerk)** | Explicitly banned by the rules. Sessions are implemented in-house. |
| **WebSockets / real-time dashboards** | Nice demo, no rubric points, meaningful complexity (ASGI, channel layer). Polling is fine for judge progress. |
| **GraphQL** | The rubric asks for REST + OpenAPI. |
| **Prometheus / Grafana stack** | Two more containers for observability nobody will operate during judging. Structured logs + `/healthz` cover the operability criterion. |

> Every row in the right-hand table is a line we can defend out loud in the demo video. "We didn't add Redis" is a stronger answer than an unused Redis container, **provided we can say why.**

---

## 2. Architecture — modular monolith

One Django project, one deployable, internally split by **domain**, not by technical layer.

```
backend/
├── config/                 # settings, root urls, asgi/wsgi
└── apps/
    ├── user/               # identity: accounts, sessions, auth
    ├── event/              # events, tracks, prizes, phase windows
    ├── team/               # teams, membership, invites
    ├── submission/         # projects, drafts, deadline enforcement, gallery
    ├── judging/            # rubrics, assignment, scores, normalization
    ├── community/          # votes, comments, abuse controls
    ├── audit/              # append-only event log
    └── core/               # shared: permissions, pagination, throttles, jobs
```

Each app owns the same internal shape:

```
models.py        # persistence only — no business rules beyond invariants
selectors.py     # read queries (indexed, select_related, no side effects)
services.py      # write operations — the ONLY place state changes
serializers.py   # transport shape validation (I/O contract, not business rules)
views.py         # HTTP concerns: auth, status codes, delegation to services
permissions.py   # role gates for this domain
tests/
```

**The rule that keeps this clean:** views never touch the ORM for writes, and services never touch `request`. A service function is callable from a view, a management command, a background job or a test with identical behaviour. This is what makes the CSV export, the seed script and the API share one code path.

### Why a monolith and not services
Judging integrity requires that assignment, scoring and normalization see a consistent snapshot. Inside one Postgres transaction that is free. Across services it requires distributed transactions or eventual consistency — strictly worse, for zero benefit at this scale.

---

## 3. Runtime topology

```
                    ┌─────────────────────────────────┐
                    │        docker compose up        │
                    └─────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌─────────────────┐         ┌────────────────┐
│  web (nginx)  │          │   app (Django)  │         │  db (Postgres) │
│               │          │                 │         │                │
│ • serves      │  proxy   │ • gunicorn      │  TCP    │ • data         │
│   built React │ ───────► │   N workers     │ ──────► │ • cache table  │
│ • /api/* pass │          │ • DRF API       │         │ • FTS index    │
│ • gzip, cache │          │ • job runner    │         │ • volume       │
│   headers     │          │ • /healthz      │         │                │
└───────────────┘          └─────────────────┘         └────────────────┘
        ▲
        │  browser: SPA + XHR to /api
     [ user ]
```

**Three containers. That is the whole system.**

| Container | Responsibility | Notes |
|---|---|---|
| `web` | TLS termination point (optional), static asset serving, single origin for SPA + API | Same-origin removes CORS from production entirely and makes cookie auth simple |
| `app` | Django under gunicorn with a small worker count; background job runner in the same image | `depends_on: db` with a health check so the first boot is not a race |
| `db` | Postgres 16 with a named volume | Migrations + idempotent seed run on app start |

**Deliberate note on "load balancing":** gunicorn runs multiple worker processes, and nginx distributes across them via a single upstream socket. That is the correct amount of load distribution for this deployment. Anything above it is unjustifiable on one host.

### Startup sequence (must be deterministic)
```
db healthy ──► migrate ──► seed (idempotent, skips if data present) ──► gunicorn ──► healthz green
```
An organizer running `docker compose up` on a clean machine with no network must land on a working, populated portal. That single property is worth more rubric points than any optimisation in this document.

---

## 4. Caching strategy

Caching is only defensible when the invalidation story is written down. Ours:

| Layer | What is cached | Backend | Invalidation | Justification |
|---|---|---|---|---|
| **L1 — HTTP** | Built JS/CSS assets | nginx + content-hashed filenames | Filename changes on build | Free, zero risk |
| **L2 — Computed results** | Leaderboard / normalized ranking per event | Postgres cache table (`django.core.cache.backends.db`) | Explicit `cache.delete()` on any score write, assignment change, or new normalization run — triggered inside the same service function | Ranking is a full-table aggregate; recomputing per page view is wasteful, and its inputs change rarely and through exactly one code path |
| **L3 — Materialized rows** | `normalized_score` rows per run | Postgres table (not a cache — persisted, versioned by `run_id`) | Never invalidated; a new run writes a new `run_id` | Makes results reproducible and auditable, which is worth more than the speed |
| **L4 — Per-request memoization** | Role lookup `(user, event) → role` | Request-local dict | Dies with the request | Role is checked on nearly every endpoint; caching it beyond a request would be an authorization bug |
| **Explicitly NOT cached** | Judge dashboards, submission detail, vote counts during voting | — | — | Freshness matters more than a few milliseconds, and stale judging data is an integrity problem |

**Rule:** anything that affects who-wins is either uncached or cached with a version key. Never a TTL guess.

**Why Postgres as the cache backend rather than Redis:** it is already running, it is transactional (a cache write can join the same transaction as the data write), it survives restart, and it costs zero extra containers. At a few thousand keys the latency difference is irrelevant.

---

## 5. Background work without a broker

Slow operations: normalization runs, CSV/JSON exports, certificate signing in bulk, webhook delivery with retries.

These must not run inside the request. But Celery + a broker is three containers for jobs that take under a second here. The middle path:

- A `job` table (`id, kind, payload, status, attempts, run_after, result, error`) — the queue lives in Postgres.
- A worker thread inside the app container polls with `SELECT ... FOR UPDATE SKIP LOCKED`, which is exactly how a real job queue avoids double-processing, and is safe across gunicorn workers.
- Webhooks use the **outbox pattern**: the row that triggers the webhook and the outbox entry are written in the *same transaction*, so a crash cannot lose a notification or send one for a rolled-back change. Retries use exponential backoff with a cap.
- The job table doubles as a visible audit surface: an organizer can see which exports ran and which webhooks failed.

This gives real queue semantics (at-least-once delivery, retries, no double-processing, crash safety) with **zero extra containers**, and it is straightforwardly explainable in the demo. If profiling ever justified Celery, the `job` table interface is the seam to swap behind.

---

## 6. Data layer rules

| Rule | Reason |
|---|---|
| Every foreign key that is filtered on gets an index; composite index where the query filters on two columns together (e.g. `(event_id, status)`) | Query plans, not vibes |
| Uniqueness is a **database constraint**, never only a serializer check | Serializer checks are check-then-act and lose the race under concurrency |
| Case-insensitive uniqueness via `UniqueConstraint(Lower("email"))` | `Test@x.com` and `test@x.com` must be one account |
| Money-free, but score arithmetic uses `numeric`, never `float` | Reproducible ranking; float summation order changes results |
| All timestamps stored UTC; per-event display timezone stored as data | Deadlines are published in UTC and participants are global |
| Soft-delete only where an audit trail requires it; otherwise hard delete | Avoids the "deleted rows still counted" class of bug |
| `select_for_update` around read-modify-write on scores and vote counters | Prevents lost updates |
| Read paths use `selectors.py` with explicit `.only()` / `select_related` | Stops N+1 before it starts |

---

## 7. Security & role isolation (the part that is actually scored)

| Control | Implementation |
|---|---|
| Default deny | `DEFAULT_PERMISSION_CLASSES = ["rest_framework.permissions.IsAuthenticated"]`; public endpoints opt in with `AllowAny` explicitly |
| Role resolution | `(user, event) → role` from the `event_role` table on every request; never from a client-supplied field |
| Object-level checks | A judge may read a submission **only** if an `assignment` row links them to it; enforced in the queryset, not with an `if` after fetching |
| Conflict of interest | Assignment algorithm refuses to assign a judge to their own team's submission; enforced again at score-write time |
| Results embargo | Score and ranking endpoints return 403 until `results_published_at`, including for the API and CSV export |
| Password policy | Django's `AUTH_PASSWORD_VALIDATORS` invoked in the serializer (configuring them in settings does nothing on its own) |
| Rate limiting | Per-scope throttles on register, login, vote, comment |
| Secrets | Environment only; `.env` never committed; container reads real env vars |
| Audit | Append-only log with actor, action, entity, before/after payload; no UPDATE/DELETE grants |

**Threat model note (bonus +3):** the attacks named in the brief — Sybil voting, ballot stuffing, submission scraping, judge collusion, deadline gaming — each map to a control above. Scraping in particular is why identifiers are UUIDs rather than sequential integers: sequential IDs let an attacker enumerate every submission by counting.

---

## 8. Performance budget (what "fast enough" means here)

Targets, measured with the fixture dataset — not aspirational numbers:

| Endpoint | Budget | Technique |
|---|---|---|
| Gallery list (paginated, 24/page) | < 150 ms | index + `select_related` + pagination |
| Gallery search | < 200 ms | GIN index on `tsvector` |
| Judge dashboard | < 150 ms | one query with aggregates, no N+1 |
| Score submit | < 100 ms | single transaction, one lock |
| Normalization run (300 projects × 50 judges) | < 5 s, backgrounded | numpy vectorised, one bulk insert |
| Full CSV export | < 3 s, streamed | server-side cursor, `StreamingHttpResponse` |

If a number is missed, the fix is a query plan, not a cache.

---

## 9. What we would do differently at real scale

Included because the ability to name the boundary is part of engineering judgment:

| At 100× this load | Change |
|---|---|
| Sustained concurrent judging | Move cache to Redis; keep the same cache-key contract |
| Job volume grows | Swap the `job` table worker for Celery behind the same service interface |
| Many concurrent events | Read replica for gallery/search traffic; writes stay on primary |
| Multi-tenant SaaS | Row-level security by event, then schema-per-tenant |
| Global audience | CDN for static assets and gallery images |

None of these are implemented, because none of them are true today. The seams (`services.py`, the cache-key contract, the `job` table) are placed so each swap is local.

---

## 10. Open items to reconcile with the published specification

1. Whether the acceptance suite drives the HTTP API directly, and what authentication it expects (session cookie vs bearer token) — this decides the auth mechanism.
2. Required normalization method(s) — per-judge z-score assumed until the spec says otherwise.
3. Fixture data shape — drives the seed script and possibly field names.
4. Whether file uploads are in scope (local volume only if so).
5. Whether one submission per team or per track.

---

## Appendix A — Configuration deltas this document implies

```python
# config/settings.py

DATABASES = {
    "default": {
        **env.db("DATABASE_URL"),
        "CONN_MAX_AGE": env.int("CONN_MAX_AGE", default=60),
        "CONN_HEALTH_CHECKS": True,
    }
}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.db.DatabaseCache",
        "LOCATION": "cache_table",
    }
}

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",   # default deny
    ],
    "DEFAULT_PAGINATION_CLASS":
        "rest_framework.pagination.LimitOffsetPagination",
    "PAGE_SIZE": 24,
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "register": "5/hour",
        "login": "10/hour",
        "vote": "60/hour",
        "comment": "20/hour",
    },
    "EXCEPTION_HANDLER": "apps.core.exceptions.handler",   # uniform error shape
}
```

---

*This document is a living design record. Every entry in §1.2 is a decision we must be able to defend, not an omission we must apologise for.*
