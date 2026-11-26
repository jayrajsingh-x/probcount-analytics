# ProbCount HTTP API Reference

Start the server:

```bash
uvicorn probcount.api:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI docs are available at `/docs`.

---

## Endpoints

| Method & Path | Auth | Description |
|---|---|---|
| `POST /events` | Optional `X-API-Key` | Ingest a single visit event (`202`) |
| `POST /events/batch` | Optional `X-API-Key` | Ingest up to 10,000 events (`202`) |
| `GET /stats/unique` | Public | Unique visitors (`site`, `page`, `start`, `end`, `window`, `compare`) |
| `GET /stats/views` | Public | Page-view count + CMS error bound |
| `GET /stats/top` | Public | Top-`k` pages in window |
| `GET /stats/series` | Public | Hourly/daily/minute unique-visitor time series |
| `GET /stats/overview` | Public | Headline KPIs |
| `GET /stats/new-vs-returning` | Public | Approximate new vs returning visitors (HLL inclusion-exclusion) |
| `GET /stats/overlap` | Public | Page visitor intersection & Jaccard similarity |
| `POST /sketch/hll/merge` | Optional `X-API-Key` | Merge a base64-encoded HLL into a named sketch |
| `GET /sketch/hll/{name}` | Public | Retrieve a named HLL sketch |
| `POST /admin/snapshot` | Optional `X-API-Key` | Force immediate snapshot save |
| `POST /admin/simulate` | Optional `X-API-Key` | Run synthetic traffic generator into engine |
| `POST /admin/reset` | Optional `X-API-Key` | Reset in-memory engine state |
| `GET /health` | Public | Health check & uptime |
| `GET /metrics` | Public | Processed/rejected/late counts, memory breakdown, sketch params |

---

## Example `curl` Commands

```bash
# Ingest a single event
curl -X POST http://127.0.0.1:8000/events \
  -H "Content-Type: application/json" \
  -d '{"visitor_id":"u_123","page_id":"/products/42"}'

# Query unique visitors
curl "http://127.0.0.1:8000/stats/unique?site=default&window=24h"

# Trigger simulation & inspect top pages
curl -X POST http://127.0.0.1:8000/admin/simulate \
  -H "Content-Type: application/json" \
  -d '{"n_events":50000,"n_visitors":10000,"n_pages":50}'
curl "http://127.0.0.1:8000/stats/top?k=5"
```

---

## Privacy by Design

1. **Edge Salted Hashing**: Raw `visitor_id` values are immediately hashed at the API edge
   using keyed `BLAKE2b` with `PROBCOUNT_HASH_SALT` before entering any sketch. Raw visitor
   identifiers are never stored in memory, logs, or snapshots.
2. **Non-Invertible Sketches**: HyperLogLog stores only `(bucket_index, max_leading_zeros)`
   and Count-Min Sketch stores aggregated counters — neither structure can reconstruct
   individual user histories.

---

## Concurrency Model

The service protects `AnalyticsEngine` with a `threading.RLock`. Because HLL/CMS updates
take microseconds in memory, lock contention is minimal and avoids the extra latency of an
async worker queue while guaranteeing zero lost updates under concurrent `POST /events/batch`
requests.
