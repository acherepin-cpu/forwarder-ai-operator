# Forwarder AI Operator

**An AI operator for parcel forwarding (Spain → Kazakhstan) that checks every order the way an experienced forwarding employee would — before the parcel gets stuck at customs.**

Hacktoberfest Hack Day Barcelona 2026 · Tracks: **Best Open-Source AI Project**, **Best Use of Gemma 4** · Theme: **expertise → AI**

## What it is and who it is for

For **mail-forwarding services** shipping online purchases from Europe to Central Asia, and for **their customers** in Kazakhstan.

The most expensive forwarding mistakes happen *before* shipping: a typo in the tracking number or national ID, a postcode from another city, a landline instead of a mobile phone, an order above the duty-free limit, a perfume that cannot fly. Today they are caught (or missed) manually, and only by experienced staff. A mistake surfaces two weeks later at customs.

The customer:

1. uploads a **screenshot of the order** (or describes it in text) → Gemma 4 extracts store, items, sizes, prices, total;
2. writes **delivery details in free form**, in one message → Gemma 4 splits them into fields;
3. gets **expert checks** and a plain-language explanation of every problem;
4. fixes mistakes in dialogue — the bot remembers which field it asked to correct, so a bare number is enough;
5. the validated order is saved under our own ID (`FWD-00001`) with status *"awaiting arrival at the warehouse"*. On arrival the parcel is matched by tracking number.

## How to run

```bash
git clone https://github.com/acherepin-cpu/forwarder-ai-operator.git
cd forwarder-ai-operator
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # put your Gemini API key and the Gemma 4 model name here
python app.py               # open http://127.0.0.1:7860
```

Get a free key at [aistudio.google.com](https://aistudio.google.com). `SHARE=1` in `.env` gives a temporary public link. `/новый` starts a new order.

**Try it (fictional data):** upload any order screenshot, then send:

> Трек RR123456785ES, Иванова Айгерим, Алматы, ул. Абая 10 кв. 5, индекс 010000, ИИН 900101100004, телефон +7 701 123 45 67

The bot reports that postcode 010000 belongs to Astana, not Almaty. Reply `050000` — the order is saved and appears in the **«База заказов»** tab.

## Expertise: what is encoded and where it comes from

All rules live in `app.py`, section **«Экспертные правила»** (`check_track`, `check_iin`, `check_postcode`, `check_phone`, `validate`). They come from the author's work launching a Spain → Kazakhstan forwarding service and from public standards.

| Check | Rule | Source of knowledge |
|---|---|---|
| Tracking number | Detects carrier (Correos, UPS, DHL, GLS/SEUR, Amazon); verifies the check digit of international postal numbers; flags incomplete numbers | UPU S10 standard, carriers' public formats |
| IIN (Kazakh national ID) | 12 digits, valid birth date in the first 6, check-digit algorithm | Kazakh IIN specification |
| Postcode | Old (6 digits) and new (A05A0K) formats; postcode must match the city | Kazpost formats (demo subset: Almaty, Astana, Shymkent, Karaganda) |
| Phone | Must be a Kazakh **mobile** number (operator codes), needed for delivery SMS | Kazakh numbering plan |
| EAEU duty-free limit | Orders above €200 → suggests splitting into several parcels | EAEU e-commerce rules (limit kept at €200 in 2026) |
| Restricted items | Perfume, aerosols, batteries, alcohol, knives… → operator review | Air-transport restrictions for parcels |
| Required data | Size only for clothing/shoes; store order number optional (we match by tracking number) | Operational practice |

**Design principle:** the LLM does what LLMs do well — reading screenshots, understanding free-form text, dialogue. The expertise is deterministic code, so checks are exact, explainable and testable.

## Model and its license

- **Gemma 4** (Google DeepMind), open-weight model, released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). Docs: [ai.google.dev/gemma](https://ai.google.dev/gemma).
- Accessed through the **Gemini API** (model name set in `.env`, `GEMMA_MODEL`), subject to the [Gemini API Terms](https://ai.google.dev/gemini-api/terms). The integration is in `app.py` → `call_model`, `recognize_order`, `extract_fields`.
- Fallbacks: other Gemma 4 sizes via the API if the main one is overloaded, and optionally a local Gemma via Ollama (`OLLAMA_MODEL`).

**Why open weights matter here:** forwarding means passport-level personal data (national IDs, addresses, phones). With an open-weight model the same pipeline can run on the company's own server so customer data never leaves it. The orders table already masks national IDs.

## Proof: better than the same model without our layer

`eval.py` runs 10 fictional delivery records with known mistakes through (a) bare Gemma 4, asked to find the wrong fields, and (b) our product. A case counts only if exactly the right fields are flagged.

```bash
python eval.py      # prints the table and writes eval_results.md
```

Results: see [`eval_results.md`](eval_results.md).

## Where it does not work (yet)

- Blurry or cropped screenshots, and stores whose order pages show prices without a total — the bot asks for the missing data instead of guessing.
- Postcode-to-city check covers only 4 cities; other cities get format checks only.
- Tracking numbers are checked for **format and check digit**, not for existence — that needs carrier APIs.
- Non-EUR orders are flagged for the operator, not converted.
- No weight check (31 kg limit) — weight is known only at the warehouse.
- Demo is single-user (draft kept in memory) with a simple global rate limit.

## Security and privacy

- API key only in `.env` (git-ignored); `.env.example` has no real values.
- Screenshot and message text are treated as data: prompts instruct the model to ignore embedded instructions, the model only returns JSON, and nothing is saved unless deterministic rules pass.
- No real personal data in examples or tests. National IDs are masked in the UI.
- Customs hints are informational, not legal advice (disclaimer in the UI).

## Roadmap

Telegram bot on the same engine · local Gemma 4 for on-premise processing · warehouse side: match incoming parcels by tracking number, photo on arrival, customer notification · carrier tracking APIs · full Kazpost directory · sanctions/dual-use screening.

## License

[MIT](LICENSE)
