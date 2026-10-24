"""
Forwarder AI Operator — шаг 2.

ИИ-оператор мейлфорвардинга (Испания → Казахстан):
1. клиент присылает скриншот заказа → Gemma 4 распознаёт товары и суммы;
2. клиент пишет трек-номер, ФИО, адрес, индекс, ИИН, телефон в свободной форме → Gemma 4 извлекает поля;
3. экспертные правила проверяют данные (трек-номер, ИИН, индекс, телефон, лимит ЕАЭС, ограниченные товары);
4. бот объясняет ошибки, клиент исправляет, корректный заказ сохраняется в SQLite.

Демо рассчитано на одного пользователя: черновик заказа хранится в памяти процесса.
"""

import io
import json
import os
import re
import sqlite3
import time
from datetime import datetime

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMMA_MODEL", "gemma-4-31b-it")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.db")
DUTY_FREE_LIMIT_EUR = 200
STATUS_AWAITING = "ожидаем поступления на склад"
STATUS_REVIEW = "ожидаем поступления на склад, нужна проверка оператора"

client = genai.Client(api_key=API_KEY)

# ---------------------------------------------------------------------------
# Промпты для Gemma
# ---------------------------------------------------------------------------

ORDER_PROMPT = """Ты — оператор сервиса мейлфорвардинга. На изображении скриншот заказа из интернет-магазина.
Извлеки данные и верни ТОЛЬКО JSON, без пояснений и без markdown, строго в таком формате:
{
  "магазин": строка или null,
  "номер_заказа": строка или null,
  "товары": [
    {
      "название": строка,
      "артикул": строка или null,
      "цвет": строка или null,
      "размер": строка или null,
      "нужен_размер": true или false,
      "цена": число или null,
      "валюта": строка или null
    }
  ],
  "итоговая_сумма": число или null,
  "валюта": строка или null
}
Правила:
- "нужен_размер": true только для одежды, обуви и аксессуаров с размером (кольца, ремни, перчатки);
  для косметики, еды, спортивного питания, электроники, книг и т.п. — false;
- цены пиши числом с точкой, например 119.00, без символа валюты;
- валюту пиши кодом: EUR, USD и т.д.;
- если каких-то данных нет на изображении, ставь null и ничего не выдумывай."""

TEXT_PROMPT = """Ты — оператор сервиса мейлфорвардинга из Испании в Казахстан.
Из сообщения клиента извлеки данные и верни ТОЛЬКО JSON, без пояснений и без markdown:
{
  "трек_номер": строка или null,
  "фио": строка или null,
  "город": строка или null,
  "адрес": строка или null (улица, дом, квартира — без города и индекса),
  "индекс": строка или null,
  "иин": строка или null,
  "телефон": строка или null,
  "номер_заказа": строка или null,
  "размер": строка или null,
  "итоговая_сумма": число или null
}
Правила:
- переписывай номера, индексы, ИИН и телефоны символ в символ, ничего не исправляй;
- если поля в сообщении нет, ставь null и ничего не выдумывай.

Сообщение клиента:
"""

# ---------------------------------------------------------------------------
# Работа с моделью
# ---------------------------------------------------------------------------


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Модель не вернула JSON")
    return json.loads(match.group(0))


def available_gemma_models() -> list[str]:
    try:
        return [m.name for m in client.models.list() if "gemma" in m.name.lower()]
    except Exception:
        return []


def call_model(contents) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL, contents=contents)
            return parse_json(response.text)
        except Exception as e:
            last_error = e
            if any(c in str(e) for c in ("500", "503", "INTERNAL", "UNAVAILABLE")):
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise last_error


def prepare_image(path: str, max_side: int = 1600) -> bytes:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def recognize_order(path: str) -> dict:
    image = types.Part.from_bytes(data=prepare_image(path), mime_type="image/jpeg")
    return call_model([image, ORDER_PROMPT])


FIELD_LABELS = {
    "трек_номер": "трек-номер", "фио": "ФИО получателя", "город": "город",
    "адрес": "адрес", "индекс": "индекс", "иин": "ИИН", "телефон": "телефон",
}


def extract_fields(text: str, awaiting: list[str]) -> dict:
    # Короткий ответ без пояснений («900101100004») относим к полю, которое бот только что просил исправить
    if len(awaiting) == 1 and re.fullmatch(r"[\w\s+\-()./]{1,40}", text) and len(text.split()) <= 5:
        if not re.search(r"[a-zа-яё]{4,}", text.lower()) or awaiting[0] in ("фио", "город", "адрес"):
            return {awaiting[0]: text}
    context = ""
    if awaiting:
        names = ", ".join(FIELD_LABELS[k] for k in awaiting)
        context = (
            f"Контекст: оператор только что попросил клиента исправить или прислать: {names}. "
            f"Если значение в сообщении не подписано, относи его к этим полям.\n"
        )
    fields = call_model([context + TEXT_PROMPT + text])
    # ИИН берём из исходного текста напрямую, чтобы модель не «поправила» цифры
    iin_match = re.search(r"(?<!\d)\d{12}(?!\d)", text.replace(" ", ""))
    if iin_match and fields.get("иин"):
        fields["иин"] = iin_match.group(0)
    return fields


# ---------------------------------------------------------------------------
# Экспертные правила
# ---------------------------------------------------------------------------

# Демо-подмножество: старые 6-значные индексы (префикс) и новые буквенно-цифровые (первая буква)
CITY_POSTCODES = {
    "алматы": ("050", "A"),
    "астана": ("010", "Z"),
    "шымкент": ("160", "X"),
    "караганда": ("100", "M"),
}
CITY_ALIASES = {
    "almaty": "алматы", "алма-ата": "алматы",
    "astana": "астана", "нур-султан": "астана",
    "shymkent": "шымкент", "чимкент": "шымкент",
    "karaganda": "караганда", "қарағанды": "караганда",
}

RESTRICTED_KEYWORDS = [
    "perfum", "parfum", "eau de", "colonia", "aerosol", "spray",
    "batería", "bateria", "battery", "power bank", "powerbank", "litio", "lithium",
    "alcohol", "vino", "wine", "cuchillo", "knife", "navaja", "mechero", "lighter", "encendedor",
    "духи", "аэрозоль", "аккумулятор", "батарея", "нож", "алкоголь", "зажигалка",
]


def s10_check_digit_ok(code: str) -> bool:
    """Контрольная цифра международного почтового номера UPU S10 (например RR123456785ES)."""
    weights = [8, 6, 4, 2, 3, 5, 9, 7]
    total = sum(int(d) * w for d, w in zip(code[2:10], weights))
    check = 11 - total % 11
    check = 0 if check == 10 else 5 if check == 11 else check
    return check == int(code[10])


def check_track(track: str):
    """Возвращает (перевозчик, ошибка, предупреждение)."""
    t = re.sub(r"\s", "", track).upper()
    if re.fullmatch(r"[A-Z]{2}\d{9}[A-Z]{2}", t):
        country = t[-2:]
        carrier = "Correos (международное)" if country == "ES" else f"почта страны {country}"
        if not s10_check_digit_ok(t):
            return carrier, f"в трек-номере {t} не сходится контрольная цифра — скорее всего, опечатка", None
        return carrier, None, None
    if re.fullmatch(r"1Z[0-9A-Z]{16}", t):
        return "UPS", None, None
    if re.fullmatch(r"TBA\d{12}", t):
        return "Amazon Logistics", None, None
    if re.fullmatch(r"PQ[0-9A-Z]{14,21}", t):
        return "Correos (внутреннее)", None, None
    if re.fullmatch(r"\d{10}", t):
        return "DHL Express", None, None
    if re.fullmatch(r"\d{11,12}", t):
        return "GLS / SEUR", None, None
    # Похоже на международный почтовый номер, но не хватает символов
    if re.fullmatch(r"[A-Z]{0,2}\d{7,11}[A-Z]{0,2}", t) and (
        re.search(r"\d[A-Z]{2}$", t) or re.match(r"^[A-Z]{2}\d", t)
    ):
        return None, (
            f"трек-номер {t} похож на международный почтовый, но неполный: нужен формат "
            f"из 2 букв, 9 цифр и 2 букв, например RR123456785ES"
        ), None
    if re.fullmatch(r"[0-9A-Z]{8,30}", t):
        return "не определён", None, f"не удалось определить перевозчика по трек-номеру {t}, оператор проверит вручную"
    return None, f"трек-номер «{track}» не похож ни на один формат испанских перевозчиков", None


def check_iin(iin: str):
    if not re.fullmatch(r"\d{12}", iin):
        return "ИИН должен состоять ровно из 12 цифр"
    month, day = int(iin[2:4]), int(iin[4:6])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return "первые 6 цифр ИИН — это дата рождения (ГГММДД), здесь она некорректна"
    d = [int(c) for c in iin]
    s = sum(d[i] * (i + 1) for i in range(11)) % 11
    if s == 10:
        w2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        s = sum(d[i] * w2[i] for i in range(11)) % 11
        if s == 10:
            return "ИИН не проходит проверку контрольной цифры"
    if s != d[11]:
        return "в ИИН не сходится контрольная цифра — скорее всего, опечатка"
    return None


def normalize_city(city: str) -> str:
    c = city.lower().replace("г.", "").replace("город", "").strip()
    return CITY_ALIASES.get(c, c)


def check_postcode(postcode: str, city: str | None):
    p = postcode.replace(" ", "").upper()
    old = re.fullmatch(r"\d{6}", p)
    new = re.fullmatch(r"[A-Z]\d{2}[A-Z]\d[A-Z]", p)
    if not (old or new):
        return f"индекс «{postcode}» имеет неверный формат: в Казахстане это 6 цифр (050000) или код вида A05A0K"
    if city:
        c = normalize_city(city)
        if c in CITY_POSTCODES:
            prefix, letter = CITY_POSTCODES[c]
            if (old and not p.startswith(prefix)) or (new and not p.startswith(letter)):
                return f"индекс {p} не относится к городу {c.title()} — проверьте индекс или город"
    return None


# Коды мобильных операторов Казахстана (Kcell, Activ, Beeline, Tele2, Altel)
KZ_MOBILE_CODES = {
    "700", "701", "702", "705", "706", "707", "708", "747",
    "750", "751", "760", "761", "762", "763", "764", "771", "775", "776", "777", "778",
}


def check_phone(phone: str):
    raw = phone.strip()
    digits = re.sub(r"\D", "", raw)
    if raw.startswith("+"):
        if not digits.startswith("7"):
            return "телефон должен быть казахстанским: начинается с +7"
        national = digits[1:]
    elif len(digits) == 11 and digits[0] in "78":
        national = digits[1:]
    else:
        national = digits
    if len(national) != 10:
        return (
            f"в номере телефона {raw} после кода страны {len(national)} цифр, а нужно 10: "
            f"формат +7 7XX XXX XX XX"
        )
    if national[:3] not in KZ_MOBILE_CODES:
        return (
            f"код {national[:3]} не относится к мобильным операторам Казахстана — "
            f"нужен мобильный номер для SMS о доставке"
        )
    return None


def order_total(order: dict):
    if order.get("итоговая_сумма") is not None:
        return float(order["итоговая_сумма"])
    prices = [i.get("цена") for i in order.get("товары") or [] if i.get("цена") is not None]
    return float(sum(prices)) if prices else None


def validate(draft: dict):
    """Возвращает (не хватает, ошибки, предупреждения, перевозчик)."""
    order, delivery = draft["order"], draft["delivery"]
    missing, errors, warnings = [], [], []
    carrier = None

    # --- заказ ---
    if not order:
        missing.append("скриншот заказа")
    else:
        # Номер заказа магазина необязателен: посылку находим по трек-номеру, свой номер присваиваем сами
        items = order.get("товары") or []
        if not items:
            missing.append("список товаров")
        for n, item in enumerate(items, 1):
            if item.get("нужен_размер", True) and not item.get("размер"):
                missing.append(f"размер для «{item.get('название') or f'товар №{n}'}»")
        total = order_total(order)
        if total is None:
            missing.append("сумма заказа")
        else:
            currency = (order.get("валюта") or "EUR").upper()
            if currency == "EUR" and total > DUTY_FREE_LIMIT_EUR:
                warnings.append(
                    f"сумма {total:.2f} € выше беспошлинного лимита ЕАЭС {DUTY_FREE_LIMIT_EUR} €: "
                    f"предлагаю разделить заказ на несколько посылок, иначе при ввозе будет пошлина"
                )
            elif currency != "EUR":
                warnings.append(f"заказ в валюте {currency}: оператор пересчитает сумму для проверки лимита")
        for item in items:
            name = (item.get("название") or "").lower()
            hit = next((k for k in RESTRICTED_KEYWORDS if k in name), None)
            if hit:
                warnings.append(
                    f"«{item.get('название')}» может быть ограничен к авиаперевозке "
                    f"(жидкости, аэрозоли, аккумуляторы и т.п.) — оператор проверит вручную"
                )

    # --- доставка ---
    labels = {
        "трек_номер": "трек-номер", "фио": "ФИО получателя", "город": "город",
        "адрес": "адрес (улица, дом, квартира)", "индекс": "индекс", "иин": "ИИН", "телефон": "телефон",
    }
    for key, label in labels.items():
        if not delivery.get(key):
            missing.append(label)

    if delivery.get("трек_номер"):
        carrier, err, warn = check_track(delivery["трек_номер"])
        if err:
            errors.append(err)
        if warn:
            warnings.append(warn)
    if delivery.get("иин"):
        err = check_iin(re.sub(r"\s", "", delivery["иин"]))
        if err:
            errors.append(err)
    if delivery.get("индекс"):
        err = check_postcode(delivery["индекс"], delivery.get("город"))
        if err:
            errors.append(err)
    if delivery.get("телефон"):
        err = check_phone(delivery["телефон"])
        if err:
            errors.append(err)

    return missing, errors, warnings, carrier


def fields_to_fix(draft: dict) -> list[str]:
    """Поля доставки, которые сейчас ошибочны (а если ошибок нет — которые не заполнены)."""
    d = draft["delivery"]
    bad = []
    if d.get("трек_номер") and check_track(d["трек_номер"])[1]:
        bad.append("трек_номер")
    if d.get("иин") and check_iin(re.sub(r"\s", "", d["иин"])):
        bad.append("иин")
    if d.get("индекс") and check_postcode(d["индекс"], d.get("город")):
        bad.append("индекс")
    if d.get("телефон") and check_phone(d["телефон"]):
        bad.append("телефон")
    if bad:
        return bad
    return [k for k in FIELD_LABELS if not d.get(k)]


# ---------------------------------------------------------------------------
# База данных
# ---------------------------------------------------------------------------


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT, status TEXT,
                store TEXT, order_number TEXT, items TEXT, total REAL, currency TEXT,
                track TEXT, carrier TEXT,
                recipient TEXT, city TEXT, address TEXT, postcode TEXT, iin TEXT, phone TEXT,
                warnings TEXT
            )"""
        )


def save_order(draft: dict, carrier: str, warnings: list[str]) -> int:
    order, d = draft["order"], draft["delivery"]
    status = STATUS_REVIEW if warnings else STATUS_AWAITING
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            """INSERT INTO orders (created_at, status, store, order_number, items, total, currency,
               track, carrier, recipient, city, address, postcode, iin, phone, warnings)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M"), status,
                order.get("магазин"), order.get("номер_заказа"),
                json.dumps(order.get("товары") or [], ensure_ascii=False),
                order_total(order), (order.get("валюта") or "EUR").upper(),
                re.sub(r"\s", "", d["трек_номер"]).upper(), carrier,
                d["фио"], d["город"], d["адрес"], d["индекс"].replace(" ", "").upper(),
                re.sub(r"\s", "", d["иин"]), d["телефон"], "; ".join(warnings),
            ),
        )
        return cur.lastrowid


def format_id(order_id: int) -> str:
    return f"FWD-{order_id:05d}"


def load_orders() -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            "SELECT id, created_at, status, store, order_number, total, currency, track, carrier, "
            "recipient, city, postcode, iin, phone, warnings FROM orders ORDER BY id DESC",
            con,
        )
    # ИИН в интерфейсе маскируем — персональные данные
    df["iin"] = df["iin"].apply(lambda x: "********" + str(x)[-4:] if x else x)
    df["id"] = df["id"].apply(format_id)
    df.columns = [
        "Наш номер", "Создан", "Статус", "Магазин", "Заказ в магазине", "Сумма", "Валюта", "Трек-номер",
        "Перевозчик", "Получатель", "Город", "Индекс", "ИИН", "Телефон", "Предупреждения",
    ]
    return df


# ---------------------------------------------------------------------------
# Диалог
# ---------------------------------------------------------------------------


def new_draft() -> dict:
    return {"order": None, "delivery": {}, "awaiting": []}


DRAFT = new_draft()


def merge_text_fields(fields: dict):
    for key in ("трек_номер", "фио", "город", "адрес", "индекс", "иин", "телефон"):
        if fields.get(key):
            DRAFT["delivery"][key] = str(fields[key]).strip()
    order = DRAFT["order"]
    if order is not None:
        if fields.get("номер_заказа"):
            order["номер_заказа"] = str(fields["номер_заказа"]).strip()
        if fields.get("итоговая_сумма") is not None:
            order["итоговая_сумма"] = fields["итоговая_сумма"]
        if fields.get("размер"):
            for item in order.get("товары") or []:
                if item.get("нужен_размер", True) and not item.get("размер"):
                    item["размер"] = str(fields["размер"]).strip()


def format_order(order: dict) -> str:
    lines = ["**Распознанный заказ**", ""]
    lines.append(f"- Магазин: {order.get('магазин') or '—'}")
    if order.get("номер_заказа"):
        lines.append(f"- Заказ в магазине: {order['номер_заказа']}")
    for n, item in enumerate(order.get("товары") or [], 1):
        price = item.get("цена")
        price_str = f"{float(price):.2f} {item.get('валюта') or ''}".strip() if price is not None else "—"
        if item.get("нужен_размер", True) or item.get("размер"):
            size_str = f"размер {item.get('размер') or '—'} | "
        else:
            size_str = ""
        color_str = f"цвет {item['цвет']} | " if item.get("цвет") else ""
        lines.append(f"- Товар {n}: {item.get('название') or '—'} | {color_str}{size_str}{price_str}")
    total = order_total(order)
    lines.append(f"- Итого: {f'{total:.2f} ' + (order.get('валюта') or 'EUR') if total is not None else '—'}")
    return "\n".join(lines)


def format_delivery(d: dict) -> str:
    if not d:
        return ""
    names = [("трек_номер", "Трек-номер"), ("фио", "Получатель"), ("город", "Город"), ("адрес", "Адрес"),
             ("индекс", "Индекс"), ("иин", "ИИН"), ("телефон", "Телефон")]
    lines = ["**Данные доставки**", ""]
    for key, label in names:
        value = d.get(key)
        if key == "иин" and value:
            value = "********" + re.sub(r"\s", "", value)[-4:]
        lines.append(f"- {label}: {value or '—'}")
    return "\n".join(lines)


def get_image_path(message: dict):
    files = message.get("files") or []
    if not files:
        return None
    last = files[-1]
    return last if isinstance(last, str) else (last.get("path") if isinstance(last, dict) else getattr(last, "path", None))


def model_error_text(e: Exception) -> str:
    text = str(e)
    if "not found" in text.lower() or "404" in text or "model name" in text.lower():
        models = available_gemma_models()
        hint = "\n".join(f"- `{m}`" for m in models) if models else "не удалось получить список"
        return f"Модель `{MODEL}` недоступна. Укажите название в .env (GEMMA_MODEL=...). Доступные модели Gemma:\n{hint}"
    return f"Не получилось обработать запрос к модели: {text}"


def chat(message, history):
    global DRAFT
    text = (message.get("text") or "").strip()
    path = get_image_path(message)

    if text.lower() in ("/новый", "/new", "новый заказ"):
        DRAFT = new_draft()
        return "Начинаем новый заказ. Пришлите скриншот заказа из магазина."

    parts = []
    try:
        if path:
            DRAFT["order"] = recognize_order(path)
            parts.append(format_order(DRAFT["order"]))
        if text:
            merge_text_fields(extract_fields(text, DRAFT["awaiting"]))
            if DRAFT["delivery"]:
                parts.append(format_delivery(DRAFT["delivery"]))
    except Exception as e:
        return model_error_text(e)

    if not path and not text:
        return "Пришлите скриншот заказа или напишите данные доставки."

    missing, errors, warnings, carrier = validate(DRAFT)
    DRAFT["awaiting"] = fields_to_fix(DRAFT)

    if errors:
        parts.append(
            "**❌ Нужно исправить**\n\n" + "\n".join(f"- {e}" for e in errors)
            + "\n\nПришлите исправленное значение, можно просто без пояснений."
        )
    if warnings:
        parts.append("**⚠️ Обратите внимание**\n\n" + "\n".join(f"- {w}" for w in warnings))

    if missing:
        parts.append(
            "**Не хватает данных**\n\n" + "\n".join(f"- {m}" for m in missing)
            + "\n\nМожно написать всё одним сообщением, в свободной форме. Например:\n"
            "_Трек RR123456785ES, Иванова Айгерим, Алматы, ул. Абая 10, кв. 5, индекс 050000, "
            "ИИН 900101..., телефон +7 701 123 45 67, размер M_"
        )
    elif not errors:
        order_id = save_order(DRAFT, carrier, warnings)
        status = STATUS_REVIEW if warnings else STATUS_AWAITING
        parts.append(
            f"✅ **Заказ {format_id(order_id)} сохранён.** Перевозчик: {carrier}. Статус: {status}.\n\n"
            "Мы сообщим, когда посылка поступит на склад.\n\n"
            "Чтобы оформить следующий заказ, напишите /новый."
        )
        DRAFT = new_draft()

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------------------------

init_db()

with gr.Blocks(title="Forwarder AI Operator") as demo:
    gr.Markdown(
        "# Forwarder AI Operator\n"
        "ИИ-оператор мейлфорвардинга Испания → Казахстан: проверяет заказ, адрес и трек-номер "
        "как опытный сотрудник, до того как посылка застрянет на таможне."
    )
    with gr.Tab("Чат с клиентом"):
        gr.ChatInterface(fn=chat, multimodal=True)
    with gr.Tab("База заказов"):
        table = gr.Dataframe(value=load_orders, interactive=False, wrap=True)
        refresh = gr.Button("Обновить")
        refresh.click(load_orders, outputs=table)

if __name__ == "__main__":
    demo.launch()
