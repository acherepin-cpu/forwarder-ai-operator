"""
Forwarder AI Operator — шаг 1.
Веб-чат на Gradio: пользователь загружает скриншот заказа,
Gemma 4 (через Gemini API) распознаёт данные, бот показывает результат,
просит дополнить недостающие поля и проверяет беспошлинный лимит ЕАЭС.
"""

import io
import json
import os
import re
import time

import gradio as gr
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMMA_MODEL", "gemma-4-31b-it")
DUTY_FREE_LIMIT_EUR = 200

client = genai.Client(api_key=API_KEY)

PROMPT = """Ты — оператор сервиса мейлфорвардинга. На изображении скриншот заказа из интернет-магазина.
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
      "цена": число или null,
      "валюта": строка или null
    }
  ],
  "итоговая_сумма": число или null,
  "валюта": строка или null
}
Правила:
- цены пиши числом с точкой, например 119.00, без символа валюты;
- валюту пиши кодом: EUR, USD и т.д.;
- если каких-то данных нет на изображении, ставь null и ничего не выдумывай."""


def parse_json(text: str) -> dict:
    """Достаёт JSON из ответа модели, даже если он обёрнут в ```json ... ```."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Модель не вернула JSON")
    return json.loads(match.group(0))


def available_gemma_models() -> list[str]:
    try:
        return [m.name for m in client.models.list() if "gemma" in m.name.lower()]
    except Exception:
        return []


def prepare_image(path: str, max_side: int = 1600) -> bytes:
    """Уменьшает большие скриншоты (Retina) и конвертирует в JPEG — так API работает стабильнее."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def recognize(path: str) -> dict:
    data = prepare_image(path)
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=data, mime_type="image/jpeg"), PROMPT],
            )
            return parse_json(response.text)
        except Exception as e:
            last_error = e
            text = str(e)
            # повторяем только временные ошибки сервера
            if any(code in text for code in ("500", "503", "INTERNAL", "UNAVAILABLE")):
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise last_error


def order_total(order: dict):
    if order.get("итоговая_сумма") is not None:
        return order["итоговая_сумма"]
    prices = [i.get("цена") for i in order.get("товары") or [] if i.get("цена") is not None]
    return sum(prices) if prices else None


def find_missing(order: dict) -> list[str]:
    missing = []
    if not order.get("магазин"):
        missing.append("магазин")
    if not order.get("номер_заказа"):
        missing.append("номер заказа")
    items = order.get("товары") or []
    if not items:
        missing.append("список товаров")
    for n, item in enumerate(items, 1):
        name = item.get("название") or f"товар №{n}"
        if not item.get("размер"):
            missing.append(f"размер для «{name}»")
        if item.get("цена") is None:
            missing.append(f"цена для «{name}»")
    if order.get("итоговая_сумма") is None:
        missing.append("итоговая сумма заказа")
    return missing


def check_rules(order: dict) -> list[str]:
    """Экспертные проверки. Сюда добавляем новые правила."""
    issues = []
    total = order_total(order)
    currency = (order.get("валюта") or "").upper()
    if total is not None:
        if currency in ("", "EUR"):
            if total > DUTY_FREE_LIMIT_EUR:
                issues.append(
                    f"Сумма заказа {total:.2f} € превышает беспошлинный лимит ЕАЭС "
                    f"{DUTY_FREE_LIMIT_EUR} €. Предлагаю разделить заказ на несколько посылок, "
                    f"иначе при ввозе придётся платить пошлину."
                )
        else:
            issues.append(
                f"Валюта заказа {currency}: пересчитаем в евро для проверки лимита "
                f"{DUTY_FREE_LIMIT_EUR} €."
            )
    return issues


def format_order(order: dict) -> str:
    lines = ["**Распознанный заказ**", ""]
    lines.append(f"- Магазин: {order.get('магазин') or '—'}")
    lines.append(f"- Номер заказа: {order.get('номер_заказа') or '—'}")
    for n, item in enumerate(order.get("товары") or [], 1):
        price = item.get("цена")
        price_str = f"{price:.2f} {item.get('валюта') or ''}".strip() if price is not None else "—"
        lines.append(
            f"- Товар {n}: {item.get('название') or '—'} | артикул {item.get('артикул') or '—'} | "
            f"цвет {item.get('цвет') or '—'} | размер {item.get('размер') or '—'} | {price_str}"
        )
    total = order_total(order)
    total_str = f"{total:.2f} {order.get('валюта') or 'EUR'}" if total is not None else "—"
    lines.append(f"- Итого: {total_str}")
    return "\n".join(lines)


def get_image_path(message: dict):
    files = message.get("files") or []
    if not files:
        return None
    last = files[-1]
    return last if isinstance(last, str) else last.get("path")


def chat(message, history):
    path = get_image_path(message)
    if not path:
        return (
            "Пришлите, пожалуйста, скриншот заказа (лучше страницу подтверждения заказа "
            "или письмо от магазина). Нажмите на значок скрепки рядом с полем ввода."
        )

    try:
        order = recognize(path)
    except Exception as e:
        text = str(e)
        if "not found" in text.lower() or "404" in text:
            models = available_gemma_models()
            hint = "\n".join(f"- `{m}`" for m in models) if models else "не удалось получить список"
            return (
                f"Модель `{MODEL}` не найдена. Укажите точное название в файле .env "
                f"(строка GEMMA_MODEL=...). Доступные модели Gemma:\n{hint}"
            )
        return f"Не получилось распознать скриншот: {text}"

    parts = [format_order(order)]

    issues = check_rules(order)
    if issues:
        parts.append("**⚠️ Проблемы**\n\n" + "\n".join(f"- {i}" for i in issues))

    missing = find_missing(order)
    if missing:
        parts.append(
            "**Не хватает данных**\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nПришлите, пожалуйста, скриншот подтверждения заказа или напишите эти данные."
        )

    if not issues and not missing:
        parts.append("✅ Заказ выглядит корректно. Следующий шаг: адрес доставки и трек-номер.")

    return "\n\n".join(parts)


demo = gr.ChatInterface(
    fn=chat,
    multimodal=True,
    title="Forwarder AI Operator",
    description="Загрузите скриншот заказа — ИИ распознает его и проверит перед отправкой в Казахстан.",
)

if __name__ == "__main__":
    demo.launch()
