# SATYA AI — Phase 2: TruthGraph Evidence Verification Engine

Evidence-first WhatsApp fact-checking backend built with FastAPI, Pydantic, SQLAlchemy, and SQLite.

SATYA AI is designed to combat viral misinformation on messaging platforms with strict adherence to evidence-based verification. In accordance with its core design principles:
- **Never fabricates sources, quotations, evidence, or fact-check results.**
- Distinguishes factual propositions from opinions, questions, and conversational remarks.
- Employs an explicit **TruthGraph Evidence Sufficiency Gate** ensuring conservative, testable verdicts.
- When evidence is insufficient, ambiguous, or unavailable, it transparently returns `cannot_confirm` rather than guessing.

---

## Architecture Overview

```
satya-ai/
├── .env.example              # Environment configuration template
├── .env                      # Local environment settings
├── .gitignore                # Git ignore rules
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Pinned dependency requirements
├── README.md                 # Project architecture, commands, and guide
├── app/
│   ├── __init__.py           # Application package & version metadata
│   ├── main.py               # FastAPI app, lifespan, CORS, middleware, routers
│   ├── config.py             # Pydantic Settings (BaseSettings)
│   ├── logging_config.py     # Structured JSON logging formatter and setup
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py         # GET /health & database connectivity probe
│   │   ├── router.py         # Aggregated API routes
│   │   ├── verify.py         # POST /api/verify endpoint
│   │   └── whatsapp.py       # GET/POST /api/whatsapp/webhook Meta Cloud API endpoints
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py           # SQLAlchemy declarative Base
│   │   ├── session.py        # SQLite engine, foreign keys, additive schema migrations
│   │   └── models.py         # ORM models: Report, Claim, Source, ClaimEvidence, ProcessedWhatsAppMessage
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── evidence_models.py# RetrievedEvidenceRecord schema
│   │   ├── fact_check_api.py # Google Fact Check Tools API client
│   │   ├── llm_client.py     # Resilient LLM client (OpenAI/Gemini/Anthropic/Generic)
│   │   ├── source_fetcher.py # Safe direct URL text passage fetcher
│   │   ├── url_normalizer.py # Canonical URL deduplication & tracking cleaner
│   │   ├── web_search.py     # Web search provider (Tavily, Serper, Brave, DDG, Mock)
│   │   └── whatsapp_client.py# Meta WhatsApp Cloud API outbound message client
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── common.py         # VerdictStatus and EvidenceStance enums
│   │   ├── evidence.py       # Pydantic schemas for Source & ClaimEvidence
│   │   ├── claim.py          # Pydantic schemas for Claim
│   │   ├── report.py         # Pydantic schemas for Report
│   │   └── verify.py         # Request/Response schemas for /api/verify
│   └── services/
│       ├── __init__.py
│       ├── claim_extractor.py   # LLM & single-claim fallback extraction
│       ├── evidence_retriever.py# Multi-source retrieval & deduplication
│       ├── evidence_analyzer.py # Passage comparison, stance & conflict detection
│       ├── verdict_engine.py    # Evidence sufficiency gate & verdict rules
│       ├── verification.py      # End-to-end TruthGraph pipeline coordinator
│       └── whatsapp_service.py  # WhatsApp payload parsing, HMAC auth, deduplication & formatting
└── tests/
    ├── __init__.py
    ├── conftest.py           # In-memory SQLite fixture and TestClient
    ├── test_database.py      # Relational persistence & foreign key tests
    ├── test_health.py        # Health endpoint tests & error handling
    ├── test_live_evidence_regression.py # Phase 2.2 evidence quality hardening tests
    ├── test_truthgraph.py    # Phase 2 TruthGraph verification engine tests
    ├── test_validation.py    # Request schema validation tests
    ├── test_verdicts.py      # Supported verdicts & cannot_confirm policy tests
    └── test_whatsapp_integration.py # Phase 3 Meta WhatsApp Cloud API integration tests
```

---

## Supported Verdicts & Sufficiency Gate Rules

SATYA AI explicitly recognizes five structured verdict statuses:
1. `verified` — Sufficient, credible primary or fact-checking evidence corroborates the core assertion without contradiction.
2. `false` — Authoritative evidence directly refutes the claim. (*Lack of search results is NEVER treated as false.*)
3. `outdated` — Claim had historical validity but is now expired, revoked, discontinued, or superseded.
4. `partly_supported` — Core assertions are substantiated, but material elements are unsupported or unverified.
5. `cannot_confirm` — Insufficient evidence, conflicting credible sources, or retrieval unavailable.

---

## Required & Optional Environment Variables

Configured in `.env` (derived from `.env.example`):

| Variable | Description | Default |
|---|---|---|
| `APP_NAME` | Name of the service | `SATYA AI` |
| `APP_ENV` | Environment (`development`, `test`, `production`) | `development` |
| `DEBUG` | Enable debug mode and `/docs` | `true` |
| `DATABASE_URL` | SQLAlchemy database URI | `sqlite:///./satya.db` |
| `LOG_LEVEL` | Log verbosity | `INFO` |
| `SEARCH_PROVIDER` | Web search provider: `duckduckgo` (free live), `tavily`, `serper`, `brave`, `mock` | `duckduckgo` |
| `SEARCH_API_KEY` | General API key for external search provider | `""` |
| `TAVILY_API_KEY` | Direct alias for Tavily search API key | `""` |
| `SERPER_API_KEY` | Direct alias for Serper Google search API key | `""` |
| `GOOGLE_FACT_CHECK_API_KEY` | Google Fact Check Tools API Key | `""` |
| `LLM_PROVIDER` | LLM provider: `openai`, `gemini`, `anthropic`, `generic`, or `none` | `none` |
| `LLM_API_KEY` | API key for the LLM provider | `""` |
| `LLM_BASE_URL` | Custom endpoint for OpenAI-compatible APIs | `""` |
| `LLM_MODEL` | Model identifier | `gpt-4o-mini` |
| `RETRIEVAL_TIMEOUT_SECONDS`| Network timeout for external calls (seconds) | `10.0` |

---

## Windows PowerShell Setup & Run Commands

### 1. Clone or Open Project Directory

```powershell
Set-Location "d:\Hack On Track"
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
# Or using uv:
# uv venv --python 3.12 .venv
```

### 3. Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

### 5. Setup Environment Configuration

```powershell
Copy-Item .env.example .env
```

### 6. Run the Development Server

```powershell
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## Running the Automated Test Suite

Execute all 28 automated unit and integration tests:

```powershell
.\.venv\Scripts\pytest -v
```

---

## TruthGraph Pipeline Execution Flow

```
User Message Text
       │
       ▼
[Claim Extractor] ── (LLM extraction or single-claim fallback)
       │
       ▼ Distinct Claims
[Evidence Retriever] ───► Google Fact Check Tools API
       │             ───► Configurable Web Search (Tavily/Serper/Brave/DDG)
       │             ───► Safe Direct Source Fetcher
       │             ───► Canonical URL Normalization & Deduplication
       ▼ Retrieved Passages
[Evidence Analyzer] ────► Stance Detection (supports / refutes / neutral / inconclusive)
       │             ───► Temporal Validity (detects outdated / expired status)
       │             ───► Source Conflict Detection
       ▼ Analyzed Relationships
[Verdict Engine] ───────► Sufficiency Gate Rules Evaluation
       │
       ▼ Persist
[SQLite Database] ──────► Report ➔ Claims ➔ Sources ➔ ClaimEvidence
       │
       ▼ Return
Structured JSON Report
```

---

## API Endpoints & Examples

### Health Check (`GET /health`)

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get | ConvertTo-Json
```

```json
{
  "status": "healthy",
  "service": "SATYA AI",
  "version": "0.1.0",
  "environment": "development",
  "database": "connected",
  "timestamp": "2026-10-12T05:10:05.734148+00:00"
}
```

### Verify Claims (`POST /api/verify`)

```powershell
$body = @{
    text = "Viral forward: Central bank is closing down all ATM networks starting tonight."
    language = "en"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/verify" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 6
```

```json
{
  "status": "completed",
  "message": "TruthGraph verification completed with verdict: cannot_confirm.",
  "report": {
    "id": "9d47390e-0447-44f6-a049-deaec50c97dc",
    "input_text": "Viral forward: Central bank is closing down all ATM networks starting tonight.",
    "language": "en",
    "overall_verdict": "cannot_confirm",
    "summary": "Insufficient authoritative evidence was found to confirm the claims.",
    "claims": [
      {
        "id": "41b13e96-be58-4434-899e-a29e63a81cb7",
        "claim_text": "Viral forward: Central bank is closing down all ATM networks starting tonight.",
        "verdict": "cannot_confirm",
        "confidence_score": 0.0,
        "explanation": "Insufficient relevant evidence found from authoritative sources. In accordance with SATYA AI's evidence-first policy, ungrounded claims are assigned a cautious verdict of cannot_confirm.",
        "claim_evidence": []
      }
    ]
  }
}
```

---

## Phase 3: Meta WhatsApp Cloud API Integration

SATYA AI integrates directly with the Meta WhatsApp Cloud API to receive user messages, evaluate claims using the TruthGraph engine, and reply with evidence-backed reports.

### Webhook Endpoints
- **GET `/api/whatsapp/webhook`**: Meta webhook handshake verification (`hub.mode`, `hub.verify_token`, `hub.challenge`).
- **POST `/api/whatsapp/webhook`**: Receives incoming user messages and status events with HMAC-SHA256 signature validation (`X-Hub-Signature-256`).

### Configuration Variables (`.env`)

| Variable | Description | Example / Default |
|---|---|---|
| `WHATSAPP_ACCESS_TOKEN` | Meta System User or Permanent User Access Token | `EAA...` |
| `WHATSAPP_PHONE_NUMBER_ID` | WhatsApp Phone Number ID from Meta Dashboard | `10987654321` |
| `WHATSAPP_VERIFY_TOKEN` | Secret verify token set in Meta Webhook config | `satya_verify_token_secret` |
| `WHATSAPP_APP_SECRET` | Meta App Secret for HMAC-SHA256 signature verification | `d41d8cd98f00b204e9800998ecf8427e` |
| `WHATSAPP_GRAPH_API_VERSION` | Meta Graph API Version | `v21.0` |

---

## Local Setup & Meta Webhook Configuration (Windows PowerShell)

Follow these exact steps to connect your local SATYA AI server to Meta WhatsApp Cloud API.

### Step 1: Start the SATYA AI Server

Open Windows PowerShell in the project directory:

```powershell
Set-Location "d:\Hack On Track"
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify that the server is healthy:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

### Step 2: Start a Public HTTPS Tunnel

Meta requires a publicly accessible HTTPS URL. Use either **ngrok** or **Cloudflare Tunnel (cloudflared)**:

#### Option A: Using ngrok
```powershell
# In a new PowerShell terminal:
ngrok http 8000
```
Note the generated forwarding URL, e.g.: `https://abc-123.ngrok-free.app`

#### Option B: Using cloudflared (Free, no account needed)
```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```
Note the generated URL, e.g.: `https://random-words.trycloudflare.com`

### Step 3: Configure Meta Developer Portal

1. Log in to [developers.facebook.com](https://developers.facebook.com/) and navigate to your App.
2. Under **WhatsApp** > **Configuration**:
   - Click **Edit** next to **Webhook**.
   - **Callback URL**: Enter `https://<your-tunnel-domain>/api/whatsapp/webhook`
     *(e.g., `https://abc-123.ngrok-free.app/api/whatsapp/webhook`)*
   - **Verify Token**: Enter the exact value configured in your `.env` for `WHATSAPP_VERIFY_TOKEN`.
   - Click **Verify and Save**.
3. Under **Webhook fields**:
   - Locate the **messages** field and click **Manage** / **Subscribe**.
   - Ensure the subscription status for `messages` is marked as **Subscribed**.

### Step 4: Configure App Secret & Access Token in `.env`

In your `.env` file, populate your actual credentials:
```env
WHATSAPP_ACCESS_TOKEN=EAAxxxxxxxxx...
WHATSAPP_PHONE_NUMBER_ID=109876543210
WHATSAPP_VERIFY_TOKEN=your_configured_verify_token
WHATSAPP_APP_SECRET=your_meta_app_secret
WHATSAPP_GRAPH_API_VERSION=v21.0
```

### Step 5: Test Webhook Handshake via PowerShell

Test the GET verification endpoint locally:
```powershell
$token = "satya_verify_token_replace_with_your_secret"
$url = "http://127.0.0.1:8000/api/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=$token&hub.challenge=987654"
Invoke-RestMethod -Uri $url -Method Get
```
Expected output:
```text
987654
```

### Step 6: Test Incoming Message Webhook via PowerShell

Simulate an incoming WhatsApp text claim:
```powershell
$payload = @{
    object = "whatsapp_business_account"
    entry = @(
        @{
            id = "WHATSAPP_BUSINESS_ACCOUNT_ID"
            changes = @(
                @{
                    field = "messages"
                    value = @{
                        messaging_product = "whatsapp"
                        metadata = @{
                            display_phone_number = "15550001234"
                            phone_number_id = "10987654321"
                        }
                        contacts = @(
                            @{
                                profile = @{ name = "Tester" }
                                wa_id = "919876543210"
                            }
                        )
                        messages = @(
                            @{
                                from = "919876543210"
                                id = "wamid.TEST_LOCAL_001"
                                timestamp = "1710000000"
                                type = "text"
                                text = @{
                                    body = "The World Health Organization confirmed the global eradication of smallpox in 1980."
                                }
                            }
                        )
                    }
                }
            )
        }
    )
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/whatsapp/webhook" -Method Post -Body $payload -ContentType "application/json" | ConvertTo-Json -Depth 6
```

### Step 7: Send a Real Test Message from WhatsApp

1. In the Meta App Dashboard under **WhatsApp** > **API Setup**:
   - Ensure your recipient phone number is added to the **To** list (for development/test numbers).
2. Open WhatsApp on your phone.
3. Send a message to the test phone number displayed in your Meta App Dashboard:
   > *"The World Health Organization confirmed global eradication of smallpox in 1980."*
4. Watch the uvicorn terminal logs:
   - Webhook POST event received.
   - TruthGraph extracts claim, retrieves WHO primary source.
   - Sufficiency gate evaluates verdict to `verified`.
   - Outbound reply dispatched to your phone with the formatted report and official WHO source link.

---

## Webhook & API Troubleshooting Guide

| Issue / Error | Root Cause | Solution |
|---|---|---|
| **Meta shows "The URL couldn't be validated"** | Callback URL is not accessible by Meta, or verify token does not match. | 1. Ensure tunnel (ngrok/cloudflared) is running.<br>2. Confirm endpoint is `/api/whatsapp/webhook`.<br>3. Verify `WHATSAPP_VERIFY_TOKEN` matches `.env`. |
| **HTTP 403: Invalid HMAC-SHA256 signature** | `X-Hub-Signature-256` does not match payload hashed with `WHATSAPP_APP_SECRET`. | Check that `WHATSAPP_APP_SECRET` in `.env` matches Meta Dashboard > App Settings > Basic > App Secret. |
| **User does not receive WhatsApp reply** | `WHATSAPP_ACCESS_TOKEN` is expired, or phone number has not opted in (for test accounts). | 1. In Meta Dashboard, generate a fresh token or use a Permanent System User Token.<br>2. Ensure recipient number is added in WhatsApp > API Setup > To. |
| **Duplicate replies sent to user** | Meta retried webhook because initial response took > 20s. | SATYA AI includes built-in message deduplication (`ProcessedWhatsAppMessage` and memory cache) to prevent re-processing identical message IDs. |
| **Audio or image messages return guidance** | SATYA AI Phase 3 supports text-based claim verification. | SATYA AI automatically responds with guidance asking users to send claims as text. |

