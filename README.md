# ElevateBox AI Voice Sales Agent

Production-ready AI outbound voice system that calls leads, qualifies them in **Telugu, Hindi or English**, classifies intent as Hot/Warm/Cold, fires a WhatsApp mid-call on high intent, books callbacks from speech, and sends a personalised follow-up with resume and architecture image.

---

## Architecture

```
TRIGGER (POST /api/v1/calls/initiate)
    │
    ▼
Exotel (+91 caller ID) ──► dials +91 8688664337
    │
    ▼ (callee answers)
ExoML bridge → Vapi SIP
    │
    ▼
Vapi Voice Pipeline
├── STT: Sarvam AI (Telugu/Hindi/English, code-mixing)
├── LLM: GPT-4o (streaming, function calling)
│     ├── classify_lead()      → Lead scorer (Hot/Warm/Cold)
│     ├── send_whatsapp()      → Meta Cloud API (MID-CALL, async)
│     ├── book_callback()      → dateparser + Google Calendar
│     └── end_call_summary()   → Post-call sequence trigger
└── TTS: Sarvam AI (female, 8kHz telephony)

POST-CALL SEQUENCE (async, after call ends)
├── GPT-4o generates personalised WhatsApp message
├── WhatsApp 1: Personalised text (exact quotes from call)
├── WhatsApp 2: Resume PDF
└── WhatsApp 3: Architecture image
```

---

## Requirements

- Python 3.11+
- Redis 7+
- Accounts: Exotel, Vapi, Sarvam AI, ElevenLabs, OpenAI, Meta WhatsApp Business, Google Cloud

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd Vox
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in all values in .env
```

Key values to configure:

| Variable | Where to get it |
|---|---|
| `EXOTEL_API_KEY` | Exotel Dashboard → Settings → API |
| `VAPI_API_KEY` | app.vapi.ai → Account → API Keys |
| `SARVAM_API_KEY` | app.sarvam.ai → API Keys |
| `OPENAI_API_KEY` | platform.openai.com |
| `ELEVENLABS_API_KEY` | elevenlabs.io → Profile → API Key |
| `WHATSAPP_ACCESS_TOKEN` | developers.facebook.com → WhatsApp → API Setup |
| `WHATSAPP_PHONE_NUMBER_ID` | Same as above |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | GCP Console → IAM → Service Accounts → Key |
| `RESUME_PDF_URL` | Host your PDF publicly (S3, Cloudflare R2, etc.) |
| `ARCHITECTURE_IMAGE_URL` | Host your architecture image publicly |

### 3. Add assets

Place your files in the `assets/` folder:
```
assets/
├── resume.pdf          ← your resume
└── architecture.png    ← your architecture diagram
```

Then host them publicly (upload to S3, Cloudflare R2, or a CDN) and set the URLs in `.env`.

### 4. Run locally with ngrok

```bash
# Terminal 1 — Start the server
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Expose to internet (use your static domain)
ngrok http --url=yourname.ngrok-free.app 8000
```

Set `APP_BASE_URL=https://yourname.ngrok-free.app` in `.env`.

### 5. Place the call

```bash
curl -X POST http://localhost:8000/api/v1/calls/initiate \
  -H "Authorization: Bearer YOUR_WEBHOOK_SECRET" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Deploy to Railway

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

Set all environment variables in the Railway dashboard under Variables.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/calls/initiate` | Place outbound call (auth required) |
| `GET` | `/api/v1/calls/{call_id}` | Get call status + lead tier |
| `POST` | `/api/v1/webhooks/vapi` | Vapi event webhook (HMAC verified) |
| `GET` | `/api/v1/webhooks/exotel/bridge/{id}` | ExoML SIP bridge |
| `POST` | `/api/v1/webhooks/exotel/status/{id}` | Exotel status callback |
| `POST` | `/api/v1/stt` | Sarvam STT (Vapi custom transcriber) |
| `POST` | `/api/v1/tts` | Sarvam TTS (Vapi custom voice) |
| `GET` | `/health` | Health check |

---

## What works / What to build next

**What works:**
- Full outbound call with Indian +91 caller ID via Exotel
- Trilingual conversation (Telugu / Hindi / English + code-switching)
- Real-time Hot/Warm/Cold classification with rule + LLM scoring
- Mid-call WhatsApp fires non-blocking while conversation continues
- Natural language callback scheduling → Google Calendar event
- GPT-4o personalised follow-up (quotes exact words from the call)
- All 4 attachments: summary, resume PDF, architecture image, developer number
- Redis-backed call state (survives restarts)
- HMAC webhook verification, rate limiting, structured logging

**What to build next:**
- CRM integration (HubSpot / Zoho) — auto-populate leads from transcripts
- Real-time lead score dashboard (WebSocket push to browser)
- Voice cloning — clone a specific brand voice for Elevate Box
- Retry logic for missed calls (no answer, busy)
- A/B testing different opening scripts

---

## Cost per call

| Component | Cost per 10-min call |
|---|---|
| Exotel telephony | ₹8–10 |
| Vapi platform | ~₹4 |
| Sarvam STT | ₹4.50 |
| GPT-4o | ~₹0.80 |
| Sarvam TTS | ₹8.40 |
| WhatsApp messages | ~₹1 |
| **Total** | **~₹27–28** |
