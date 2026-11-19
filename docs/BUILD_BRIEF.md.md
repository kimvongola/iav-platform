# Build Brief for Claude Code — Industrial AI Vision Platform (CPU-Only) + Test Console

You are building the **Industrial AI Vision Platform** defined in **SRS-IAV-001 v1.0 (Scope Baseline)**, plus a **web Test Console** used only to exercise and demonstrate the platform. Read this entire brief before writing code. Work in phases (section 9), keep tests green at the end of every phase, and keep `docs/PROGRESS.md` updated so work can resume at any point.

---

## 0. Ground rules

1. **The SRS is the source of truth.** Every SRS requirement ID (3.1.1 … 3.10.5, plus the NFR table) must map to code and to at least one automated test. Keep that mapping in `docs/traceability.md` (columns: SRS ID, requirement, module/file, test name, status).
2. **No scope creep in the platform.** The only additions beyond the SRS are:
   - the **Test Console** UI (section 7);
   - the **dev and test tools** (section 8);
   - small **"derived" requirements** without which an SRS item cannot work. Mark these `DERIVED` in the traceability file and explain why each one is needed.
3. **The Test Console is not the product HMI.** The SRS says a graphical operator HMI is out of scope, so the console is a test and demo tool.
   - It talks to the platform **only through the same public REST APIs** a customer would use: no private endpoints and no direct database access. Any gap the console exposes is therefore a real API gap.
   - It ships as an optional, separate Docker Compose profile. Every page shows the banner "Test Console — not a production HMI".
4. **No authentication, RBAC or user management** (out of scope per SRS §2). The APIs are meant to run on a trusted management network.
   - Make listen addresses and CORS origins configurable.
   - Document in the README that the services must not be exposed beyond that network.
5. **Licensing:** use only permissive licences (MIT, BSD, Apache-2.0). **Do not use Ultralytics or any AGPL/GPL code or weights.** YOLOX (Apache-2.0) is the required detector.
6. **CPU only.** No CUDA dependencies anywhere: install CPU wheels of PyTorch and ONNX Runtime.
7. **Record open questions instead of guessing.** If the SRS is ambiguous, write the question in `docs/open-questions.md` with the default you chose, make the behaviour configurable, and move on. Seed that file with the questions in section 10.

---

## 1. Target architecture (SRS §1)

There are two independently deployable stacks that share no runtime dependency (SRS 3.10.2, 3.10.3):

```
MODEL CENTER (Central Platform)                 EDGE RUNTIME (one per camera)
Dataset Mgmt → Annotation → CPU Training        RTSP → FFmpeg decode → ONNX Runtime CPU
→ ONNX Export → Model Bundle (signed)           → Tracking → Polygon Zone (+ PPE rule)
PostgreSQL + file store                         → Event Store (SQLite) + Snapshots
                                                → Transactional Outbox → HTTPS Webhook
        └── operator-controlled bundle transfer (upload) ──┘   (no automatic sync)
```

- **Platform:** Linux x86-64, Docker / Docker Compose. Each stack has its own `docker-compose.yml`.
- **Model Center:** PostgreSQL holds metadata. Datasets, checkpoints and artifacts live in a mounted volume.
- **Edge:**
  - SQLite in WAL mode stores events, tracks, outbox and config. Snapshots live in a mounted volume.
  - **One RTSP stream per Edge instance** (SRS 3.5.1). Several cameras mean several Edge instances, and the compose file must make running N instances easy.
- **Edge independence:** Edge must keep running when Model Center is down (SRS 3.10.3, NFR Availability).

### Repository layout

```
/model-center       FastAPI service + job worker (Python 3.11)
/edge               FastAPI control API + pipeline worker processes (Python 3.11)
/shared             bundle format, schemas, signing/hash utilities (pip package used by both)
/test-console       React + TypeScript + Vite UI (optional profile)
/tools              rtsp-simulator, webhook-receiver, sample-data generator, e2e scripts
/docs               traceability, open questions, ADRs, operations manual, API notes
```

---

## 2. Technology choices

| Concern | Choice |
|---|---|
| API | FastAPI + Pydantic v2; OpenAPI at `/docs` and `/openapi.json` on both stacks; versioned paths `/api/v1/...` |
| Model Center DB | PostgreSQL 16, SQLAlchemy 2.x, Alembic migrations |
| Edge DB | SQLite (WAL, `synchronous=FULL` for the outbox), SQLAlchemy 2.x, Alembic |
| Job queue (Model Center) | PostgreSQL-backed queue (`SELECT … FOR UPDATE SKIP LOCKED`) with a separate worker container. **Default concurrency = 1** for heavy jobs, configurable. Jobs: frame extraction, training, ONNX export/validation, bundle build. Statuses: `queued`, `running`, `completed`, `failed`, `cancelled`. |
| Detection training | **YOLOX-Nano** (official YOLOX, Apache-2.0), PyTorch CPU, from scratch by default (SRS 3.3) |
| Classification | A small torchvision CNN (e.g. MobileNetV3-Small or ResNet18), ONNX-exportable |
| Segmentation | A light CPU segmentation network (e.g. DeepLabV3-MobileNetV3 or LR-ASPP from torchvision), ONNX-exportable |
| Anomaly detection | A PatchCore/PaDiM-style model on a ResNet18 backbone, or a small convolutional autoencoder. Must export to ONNX; any extra state (e.g. a memory bank) is stored inside the bundle. Record the choice in an ADR. |
| Inference | ONNX Runtime CPU, configurable `intra_op_num_threads` |
| Video | FFmpeg (subprocess) for RTSP decode and for frame extraction |
| Tracking | ByteTrack-style IoU/Kalman tracker implemented in-repo (no GPL code) |
| Signing | Ed25519 via the `cryptography` package; SHA-256 for integrity |
| UI | React 18 + TypeScript + Vite, TanStack Query, API client generated from OpenAPI (`openapi-typescript`), Konva (or similar) for drawing boxes and polygons, a lightweight chart library for training curves |
| Tests | pytest, httpx, Playwright for console smoke tests |

**Pretrained weights:** fine-tuning from a pretrained checkpoint is allowed only if the checkpoint is registered with **provenance and usage-rights metadata and an approval flag** (SRS 3.3 note). Training from scratch must always work without any downloaded weights.

---

## 3. Model Center — functional requirements

### 3.1 Dataset management

| SRS ID | Implementation |
|---|---|
| 3.1.1 | CRUD `/datasets`. Deleting a dataset also deletes its images, annotations and files (soft-delete with a purge job is acceptable). |
| 3.1.2 | CRUD `/datasets/{id}/classes`: unique names, stable integer class IDs (used for the YOLO export), colour. A class still referenced by annotations cannot be deleted unless `force=true`. |
| 3.1.3 | Upload JPG/PNG, single or multipart batch. Validate by decoding the content, not by file extension. Store width and height. |
| 3.1.4 | Upload MP4/H.264 and verify the codec with ffprobe. |
| 3.1.5 | Frame-extraction job with a configurable rate (fps or every N seconds). Extracted frames become dataset images that remember their source video and timestamp. |
| 3.1.6 | Validation and **configurable limits**. Defaults: max image 20 MB, max video 10 min and 1 GB, max 1,000 images per dataset, allowed formats JPG/PNG/MP4(H.264). Reject with clear 4xx errors (include limit name, value and actual). Expose current limits at `GET /limits`; update them with `PUT /limits`. |

### 3.2 Annotation

| SRS ID | Implementation |
|---|---|
| 3.2.1–3.2.2 | CRUD bounding boxes (stored as normalised xywh plus a pixel view), each linked to a class. |
| 3.2.3 | Per-image annotation status: `unlabeled`, `in_progress`, `submitted`, `approved`, `rejected`, with reviewer name and timestamp. Only `approved` images are exported. |
| 3.2.4 | **AI-assisted labeling via REST.** `POST /images/{id}/suggestions?bundle_id=` runs a selected approved model bundle and stores the results as `suggested` annotations. A suggestion becomes a real annotation **only after the user accepts it** (accept / edit / reject endpoints, single or batch). Fully automatic labeling without confirmation is excluded. |
| 3.2.5 | Export approved annotations in YOLO dataset format (images/, labels/, data.yaml, train/val split with a seed) as a downloadable zip and as the input to training jobs. |
| DERIVED | SRS 3.3.3 requires four task types, but 3.2 only defines bounding boxes. Add the **label types the other tasks need**: image-level class (classification), polygon/mask (segmentation), and normal/anomaly flag (anomaly detection). Use a polymorphic annotation table keyed by task type and export each type in a documented format. Mark these `DERIVED` and add the question to open-questions. |

### 3.3 Training and evaluation

| SRS ID | Implementation |
|---|---|
| 3.3.1 | `POST /training-jobs` returns a job ID immediately; work runs in the worker. Cancellation is supported. |
| 3.3.2 | Before training, validate dataset structure, class mapping, empty or corrupt labels, and train/val split sizes. Return a validation report; refuse to start on a blocking error. |
| 3.3.3 | Task types: `detection` (YOLOX-Nano), `classification`, `segmentation`, `anomaly`. |
| 3.3.4 | Hyperparameters: epochs, batch size, image size, learning rate, CPU threads, early stopping (patience and metric). Each has a default and supported min/max limits enforced by the API. |
| 3.3.5 | Per-epoch status, losses and validation metrics stored and exposed (`GET /training-jobs/{id}/metrics`) and streamed with SSE for live charts. Use task-appropriate metrics: mAP@0.5 and mAP@0.5:0.95 for detection, accuracy/F1 for classification, mIoU for segmentation, AUROC for anomaly. |
| 3.3.6 | Save native PyTorch checkpoints (best and last, plus periodic ones) and artifacts (config, logs, metrics). **Resume from the last checkpoint** after a worker restart. |
| 3.3.7 | Export to ONNX (fixed opset, documented input/output shapes and preprocessing). |
| 3.3.8 | Validate the ONNX file: checker, ONNX Runtime CPU load, a dummy inference, output-shape check, and an optional parity check against PyTorch on N samples within a tolerance. Store the validation report. |

### 3.4 Model artifacts and bundles

| SRS ID | Implementation |
|---|---|
| 3.4.1 | List and download artifacts (ONNX, checkpoints, reports). |
| 3.4.2 | **Versioned bundle** (`.tar.gz`) containing `model.onnx`, `metadata.json` (bundle_id, semantic version, task type, input size, preprocessing, class map, training config, metrics, validation report, source dataset version, created_at) and `manifest.json` (SHA-256 of every file). |
| 3.4.3 | SHA-256 verification on download and on install. |
| 3.4.4 | `POST /bundles/{id}/approve` (approver name and comment) signs the manifest with the Model Center Ed25519 private key and writes `signature.sig`. Only approved and signed bundles can be installed on Edge. Key generation and rotation go through a CLI; the public key is distributed to Edge by configuration. |
| 3.4.5–3.4.6 | Implemented on Edge (section 4.6). |
| Note | Model transfer is operator-controlled: download from Model Center, then upload to Edge. **No automatic sync.** |

---

## 4. Edge Runtime — functional requirements

### 4.1 Pipeline structure

- **Separate processes:** `ingest` → `inference` → `tracking + zone + rules` → `event writer`, plus a `dispatcher` for webhooks.
- **Bounded queues** between stages with a **drop-oldest, keep-latest** policy, so a slow stage never makes the system lag behind live video (NFR Performance).
- **Configurable processing FPS** (NFR Performance). Count dropped frames and expose them in status.

### 4.2 Acquisition and inference (SRS 3.5)

| SRS ID | Implementation |
|---|---|
| 3.5.1–3.5.2 | One RTSP URL per instance. FFmpeg decodes it into raw frames (configurable resize). Settings include TCP/UDP transport and timeouts. |
| 3.5.3–3.5.4 | ONNX Runtime CPU on the active detection bundle. Output per detection: class, confidence and pixel bbox. |
| 3.5.5 | Confidence threshold (global and per class) and a class filter are configurable through the API and take effect without a restart. |
| 3.5.6 | Detect interruption (no frame for T seconds or FFmpeg exit) and reconnect automatically with exponential backoff and jitter. Log the reconnect count and the last error. |
| Note | Frames with zero detections are normal and must never count as inference failures. |

### 4.3 Tracking (SRS 3.6)

| SRS ID | Implementation |
|---|---|
| 3.6.1–3.6.2 | Tracker associates detections across frames. Track IDs are unique within a **tracking-session namespace** (a new session ID is created on every runtime start), so IDs never collide across restarts. |
| 3.6.3 | Record sampled trajectories (bottom-center points at a configurable interval, capped length) in SQLite. |
| 3.6.4 | Track states `active`, `lost` (unmatched but within the TTL) and `expired`, with configurable TTLs. |

### 4.4 Polygon zone and violation rule (SRS 3.7, 3.9.5–3.9.6)

| SRS ID | Implementation |
|---|---|
| 3.7.1 | **One normalised polygon zone per camera instance** (0–1 coordinates), validated (≥ 3 points, not self-intersecting) and editable at runtime. |
| 3.7.2 | Membership is tested on the **bottom-center point** of the tracked object's bbox. |
| 3.7.3–3.7.4 | Confirmed `zone_entry` and `zone_exit` events. |
| 3.7.5 | **Transition stabilisation**: a state change is confirmed only after it holds for N frames or T seconds. **Duplicate suppression**: no repeat event of the same type for the same track within a cooldown. All values are configurable. |
| Rule | Track loss, stream disconnection and restart **must not** create zone-exit events on their own. |
| DERIVED — rule | 3.9.5 requires a "violated rule" and "missing classes", so each zone has a **rule**: a `subject_class` (e.g. `person`) and a list of `required_classes` (e.g. `helmet`, `vest`).<br>- A required item belongs to a subject when the item's bbox overlaps the subject's bbox according to a configurable association method (default: the item's center lies inside the subject's bbox, with an optional vertical region per class, e.g. top 35 % for a helmet).<br>- A **violation** starts when a confirmed subject inside the zone has been missing ≥ 1 required class for the stabilisation window. It ends when the subject complies or leaves the zone.<br>- Mark as `DERIVED` and list in open-questions. |

### 4.5 Events and snapshots (SRS 3.8)

| SRS ID | Implementation |
|---|---|
| 3.8.1 | Event types: `zone_entry`, `zone_exit`, `violation_start`, `violation_end`, and optionally `detection` (throttled and off by default, configurable). |
| 3.8.2 | Tracks and events are stored in SQLite inside a transaction. |
| 3.8.3 | `GET /events` with filters (type, class, track_id, time range, has_snapshot) and cursor pagination. `GET /events/{id}` returns a single event. |
| 3.8.4 | Committed events survive restarts (WAL, no in-memory-only state for committed events). |
| 3.8.5 | At `violation_start`, save a **JPEG snapshot** with the subject's bbox, the missing classes and its **trajectory polyline drawn on the image**, under a generated `snapshot_id`. Endpoints: `GET /snapshots/{id}` (image) and `GET /snapshots/{id}/meta`. Snapshot retention is configurable (default: keep, warn when disk use reaches a threshold). |

### 4.6 Model install and activation (SRS 3.4.5–3.4.6)

- **Install:** `POST /models` uploads a bundle. Edge verifies the SHA-256 manifest **and** the Ed25519 signature with the configured public key, and refuses tampered or unsigned bundles.
- **Activate:** `POST /models/{id}/activate` loads the model in a staging session and runs a validation (dummy inference plus I/O and class-map checks). Only after that passes does it hot-swap into the pipeline.
- **Rollback:** if validation fails, **restore the previously active model** automatically and report the reason.
- **History:** keep the install and activation history.
- **Task-type check:** only `detection` bundles can be activated in the Edge pipeline. Reject other task types with a clear error.

### 4.7 Notifications and alarms (SRS 3.9)

| SRS ID | Implementation |
|---|---|
| 3.9.1 | One webhook destination per Edge instance: URL, timeout, TLS verify (custom CA bundle allowed), HMAC secret. |
| 3.9.2 | Filter notifications by event type and object class. |
| 3.9.3 | **Transactional outbox**: the event row and its outbox row are written in the **same SQLite transaction**. A dispatcher process delivers rows in order. |
| 3.9.4 | Delivery status per notification (`pending`, `delivering`, `delivered`, `failed`, `dead_letter`), full attempt history (time, HTTP status, latency, error) and a dead-letter list. Endpoint to **replay** a dead-letter item. |
| 3.9.5 | The **alarm payload** must contain: `notification_id` (stable UUID), `event_id`, `edge_instance_id`, `camera_id`, **entry time**, **violation time**, **violated rule** (id + name), **missing classes**, **snapshot_id**, track_id, session_id, and a payload schema version. Publish a JSON Schema for it in `/docs`. |
| 3.9.6 | Exactly **one alarm per violation event**. If a subject leaves and re-enters and violates again, that is a **new** violation and a new alarm. |
| 3.9.7 | Configurable retry policy (max attempts, exponential backoff, max delay). 4xx responses other than 408 and 429 go straight to dead-letter. |
| Delivery | At-least-once. Requests are signed with HMAC-SHA256 over `timestamp + "." + body` (headers `X-IAV-Timestamp`, `X-IAV-Signature`). HTTPS with TLS verification is on by default (NFR Notification Security). Allow `http://` only when an explicit dev flag is set. |

### 4.8 Operations (SRS 3.10, NFR)

- **Health and status:** `GET /health` (liveness) and `GET /status` on both stacks (SRS 3.10.4).
  - Camera status: connected, last frame time, fps in/out, reconnect count.
  - Inference: active model, latency p50/p95, dropped frames, queue depths.
  - Resources: CPU %, RAM, disk free.
  - Dispatcher: backlog, oldest pending age.
- **Restart and recovery:** Docker healthchecks and `restart: unless-stopped` on both stacks. Every service must recover cleanly when killed at any point (SRS 3.10.5): outbox rows are never lost, training resumes from a checkpoint, and no partial bundles remain.
- **Backup and restore:** Edge backup/restore API and CLI covering a consistent SQLite backup (online backup API), config, installed bundles and snapshots (optional flag), packaged as a `.tar.gz` with a manifest and SHA-256 (SRS §2 Backup/restore).
- **Logging:** structured JSON logs with request IDs on both stacks, written to stdout.

---

## 5. API conventions

- **Base path:** `/api/v1` on both stacks. Use consistent error bodies: `{error: {code, message, details}}`.
- **Pagination:** cursor-based. **Long operations** return a job object with polling and SSE progress.
- **OpenAPI:** every endpoint is fully described, with response models and examples. Export the specs to `docs/openapi-model-center.json` and `docs/openapi-edge.json` in CI, and generate the Test Console client from them.
- **Idempotency:** support an `Idempotency-Key` header on create endpoints that may be retried (uploads, job creation).

---

## 6. Non-functional requirements checklist

| NFR | How to satisfy and test it |
|---|---|
| Platform: Linux x86-64 | Base images on Debian or Ubuntu x86-64; no other architectures are required |
| Compute: CPU-only | No CUDA packages; CI runs on CPU |
| Deployment | Two independent Compose projects; the Test Console is a third, optional one |
| Availability | Integration test: stop Model Center and confirm Edge keeps detecting and alarming |
| Performance | Configurable FPS, bounded queues, a load test that reports achieved FPS and latency on the target CPU |
| Security | Configurable bind address and CORS; README warning about the trusted network |
| Model security | Tests for a tampered file, an unsigned bundle, a wrong key and a valid bundle |
| Notification security | Tests for HMAC verification, TLS verify on/off, and http rejected unless the dev flag is set |
| Persistence | Kill-test: kill the Edge processes mid-burst and confirm no committed event or outbox row is lost |

---

## 7. Test Console (web UI for testing — not the product HMI)

- **Build:** React + TS + Vite, served by nginx in `test-console/docker-compose.yml`.
- **Configuration:** Model Center and Edge base URLs (several Edge instances can be added by URL and selected from a dropdown).
- **Banner:** a fixed banner on every page reading "Test Console — not a production HMI".
- **API use:** only public APIs.
- **States:** clear loading, empty and error states, and readable API error messages.

### Model Center pages

1. **Datasets:** list, create, edit, delete; class manager (name, colour, ID); limits panel (view and edit `/limits`).
2. **Upload:** drag-and-drop images and videos with per-file progress and validation errors; frame-extraction form (rate) showing the job's progress.
3. **Annotate:** canvas editor for boxes (draw, move, resize, class picker, keyboard shortcuts), plus polygon, image-class and normal/anomaly modes for the other task types.
   - **AI-assist** button: choose an approved bundle, show suggestions as dashed boxes, accept/edit/reject one at a time or in bulk.
   - Image status workflow: submit, approve, reject (with a reviewer name field).
4. **Export:** YOLO export with split ratio and seed, then download.
5. **Training:** new-job form (task type, dataset, hyperparameters with limits shown); job list with statuses; job detail with **live loss and metric charts** (SSE), logs, cancel, and resume state.
6. **Models:** ONNX export and validation report, artifacts download, **bundle builder**, **approve and sign** (approver name), bundle download with SHA-256 shown.
7. **Jobs:** the full queue view across job types.

### Edge pages

1. **Status dashboard:** camera, inference, resource and dispatcher status (auto-refresh), reconnect history.
2. **Camera:** RTSP URL, transport and timeouts; test-connection button.
   - **Live preview** with overlays: boxes, track IDs, trajectories, the zone polygon, and subjects in violation highlighted.
   - The preview uses a **debug MJPEG/WebSocket endpoint** on Edge that is disabled by default and enabled by config. Mark it `DERIVED (test only)`.
3. **Zone and rule editor:** draw or edit the normalised polygon on a captured frame; edit the rule (subject class, required classes, association method); edit the stabilisation, cooldown, threshold and class-filter settings.
4. **Models:** upload a bundle, see the verification result (hash and signature), activate, see the activation report, see the rollback history.
5. **Events:** filterable, paginated table (type, class, track, time); event detail with the **snapshot image and drawn trajectory**; a link to the related alarm delivery.
6. **Notifications:** webhook configuration (URL, secret, TLS, filters, retry policy); delivery list with status; per-notification attempt history; dead-letter list with **replay**; a "send test notification" button.
7. **Backup/restore:** create and download a backup, upload and restore one, with a confirmation step.

### Console smoke tests

Use Playwright to cover the golden path: create dataset → upload → annotate → train for a few epochs on tiny data → bundle → approve → install on Edge → activate → see live detections → trigger a violation with the simulator → see the event, the snapshot and the delivered alarm.

---

## 8. Dev and test tools (`/tools`)

- **RTSP simulator:** a MediaMTX container plus FFmpeg looping a sample MP4 as an RTSP stream, so Edge can be tested without a camera. Include a script that can cut and restore the stream to test reconnect (3.5.6).
- **Webhook receiver:** a small FastAPI app that verifies the HMAC signature and timestamp window, de-duplicates by `notification_id`, stores received alarms, and has a simple page listing them. A switch makes it return 500, 429 or 400, or a timeout, to test retry and dead-letter behaviour.
- **Sample data generator:** synthetic images or a script that downloads a small, **permissively licensed** public dataset and converts it to the platform's format. Record each dataset's licence in `docs/data-licenses.md`. No AGPL or "non-commercial only" data.
- **E2E script:** `tools/e2e.sh` brings up all stacks and runs the full golden path through the APIs, exiting non-zero on any failure.

---

## 9. Phases (finish each with passing tests and an updated PROGRESS.md and traceability)

| Phase | Work |
|---|---|
| 0 | Repo scaffold, Compose files, CI (lint with ruff and mypy, pytest, OpenAPI export, console build), `/health` on both stacks, docs skeleton, open-questions seeded |
| 1 | Model Center datasets, classes, uploads, validation, limits, frame-extraction job, job queue |
| 2 | Annotation (all label types), review workflow, YOLO export |
| 3 | Detection training (YOLOX-Nano CPU): metrics, SSE, checkpoints and resume, ONNX export and validation |
| 4 | Classification, segmentation and anomaly training and export (the same pipeline contracts) |
| 5 | Bundles, SHA-256, approval and Ed25519 signing, AI-assisted labeling using bundles |
| 6 | Edge ingest (with reconnect), inference, bounded queues, FPS control, tracking, status endpoint |
| 7 | Zones, stabilisation and suppression, PPE rule, events, snapshots with trajectory, model install, activation and rollback |
| 8 | Outbox dispatcher, HMAC/TLS, retry policy, dead-letter and replay, alarm payload schema; backup/restore |
| 9 | Test Console (all pages) and Playwright smoke tests |
| 10 | Hardening: kill tests, Model Center-down test, load and FPS report on the target CPU, docs (README, operations manual, API guide, ADRs), final traceability with every SRS ID covered |

**Definition of done:**

- Every SRS ID in `docs/traceability.md` is marked implemented and tested.
- `tools/e2e.sh` passes from a clean checkout with `docker compose up`.
- No AGPL/GPL dependencies; generate a licence report with `pip-licenses` and `license-checker`.
- The operations manual explains install, configuration, model transfer, backup/restore and troubleshooting.

---

## 10. Seed `docs/open-questions.md` with these items (implement the default, keep it configurable)

1. **PPE rule association:** how a required item (helmet, vest) is matched to a person. Default: item center inside the person's bbox, with a per-class vertical region.
2. **Number of zones:** the SRS allows one zone per camera. Confirm this is enough, because the customer may need several zones.
3. **Annotation formats for non-detection tasks:** polygons, image class and normal/anomaly are DERIVED from 3.3.3. Confirm the formats.
4. **Event types emitted by default:** is the `detection` event needed, or only zone and violation events?
5. **Default values:**
   - stabilisation window: 1 s;
   - duplicate cooldown: 30 s;
   - retry policy: 10 attempts, backoff 2 s → 5 min;
   - processing FPS: 5;
   - snapshot retention.
6. **Dataset limits:** the default limits (1,000 images, 10-minute video) come from the customer's earlier requirement. Confirm them.
7. **Webhook security:** must HTTPS + HMAC be mandatory in production, or can the customer's receiver use plain HTTP during the pilot?
8. **Edge hardware:** the target CPU model, so performance can be tested, and the expected cameras per host (one Edge instance per camera).
