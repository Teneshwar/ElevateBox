# Vox_Qualifier
An autonomous outbound AI voice agent built to qualify e-commerce leads, dynamically adapt across Telugu, Hindi, and English, handle real-time code-switching, execute mid-call WhatsApp actions, and schedule callbacks from conversational context.

## 📌 Project Context & Current Scope

We are building a end-to-end voice sales agent capable of initiating outbound calls to prospective clients (e.g., target number: `8688664337`), conducting real-time discovery, and driving conversions without manual intervention.

### Core Capabilities Being Built:
1. **Automated Dialing & Voice Handling:** Self-initiated outbound dialing with low latency and human-like natural cadence.
2. **Multilingual & Code-Switching STT/TTS:** Dynamic recognition and synthesis across Telugu, Hindi, and English—including mixed-language phrases.
3. **Conversational Discovery:** Gathering customer inputs naturally (budget, product count, timeline, target features).
4. **Intent Classification & Routing:** Classifying callers into **Hot**, **Warm**, or **Cold** buckets based on indirect and direct verbal cues.
5. **Mid-Call Asynchronous Action:** Triggering instant WhatsApp messages during active intent spikes before the call finishes.
6. **Context-Aware Follow-Up & Scheduling:** Parsing natural time expressions (e.g., *"call me back tomorrow morning"*) into structured calendars and dispatching personalized WhatsApp summaries, architecture diagrams, and resume artifacts.

---

## 🛠 Status & Next Steps

- [x] Architecture & Flow Design
- [ ] Voice Orchestration & Telephony Pipeline Setup (Vapi / Retell / Twilio)
- [ ] STT Engine Configuration (Code-switching support)
- [ ] Intent & State Machine Logic (LLM Agent Prompts)
- [ ] Async WhatsApp Mid-Call Trigger Integration
- [ ] Time Parsing & Callback Scheduler Engine
- [ ] End-to-End Testing & Prototype Delivery

---

*Status: Active Development*
