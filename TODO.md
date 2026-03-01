# TODO (Next Version)

## Core architecture and config
- [ ] Unify settings into one typed config layer (prefer `pydantic-settings`) used by generation, extraction, judge, and orchestrator.
- [ ] Add a single structured logging setup with `generation_job_identifier` correlation in every stage.
- [ ] Define one stable pipeline result schema (dataclass or pydantic model) for API responses and orchestration JSON output.

## Input validation and safety
- [ ] Add a dedicated upload/input validation module:
  - max file size enforcement,
  - MIME allowlist,
  - content sniffing/verification,
  - image dimension guardrails before generation.
- [ ] Enforce path safety consistently across all stages (no path traversal, storage-root constrained reads/writes).

## Storage and job state
- [ ] Add a shared storage repository abstraction for all stages:
  - generated stable identifiers,
  - deterministic directory layout,
  - safe relative-path serving.
- [ ] Persist pipeline job state (JSON sidecar or DB):
  - stage progress events,
  - terminal status,
  - failure reason,
  - selected output metadata.

## Extraction and candidate quality
- [ ] Add pre-judge frame quality filtering before scoring:
  - blur/sharpness threshold (Laplacian variance),
  - perceptual hash deduplication,
  - optional face-presence hook.
- [ ] Keep extraction output contract unchanged for judge compatibility (stable sequence ids + metadata).

## Judge robustness
- [ ] Add retry/backoff and timeout handling for OpenAI judge requests.
- [ ] Harden response parsing with strict schema checks and safe fallback parsing.
- [ ] Add minimum score threshold policy and explicit rejection reporting in decision payload.

## Orchestrator and API readiness
- [ ] Wrap the library-first orchestrator behind one backend endpoint/job runner (no microservice split yet).
- [ ] Return pollable status payloads (`pending/running/completed/failed`) with `no-store` caching policy.
- [ ] Emit one canonical summary JSON per job for frontend consumption.

## Testing
- [ ] Add orchestrator integration test with mocked generation + real extraction fixture.
- [ ] Add OpenAI judge parser edge-case unit tests (verbose output, malformed JSON, range errors).
- [ ] Add storage/path traversal regression tests at orchestrator boundaries.
