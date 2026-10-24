"""
Сравнение: «голая» Gemma 4 против Forwarder AI Operator (Gemma 4 + экспертные правила).

10 обезличенных наборов данных доставки, у каждого известно, какие поля содержат ошибку.
- Голая Gemma получает данные и просьбу найти ошибочные поля.
- Наш продукт проверяет те же данные экспертными правилами из app.py.
Кейс засчитывается, если найден ровно правильный набор ошибочных полей.

Запуск:  python eval.py   → печатает таблицу и сохраняет её в eval_results.md
"""

import json
import time

import app
from app import FIELD_LABELS, MODEL, check_iin, check_phone, check_postcode, check_track

MODELS_USED: list[str] = []

# Все данные вымышленные
VALID = {
    "трек_номер": "RR123456785ES",
    "фио": "Иванова Айгерим",
    "город": "Алматы",
    "адрес": "ул. Абая 10, кв. 5",
    "индекс": "050000",
    "иин": "900101100004",
    "телефон": "+7 701 123 45 67",
}

CASES = [
    ("Все данные корректны", {}, []),
    ("Опечатка в последней цифре ИИН", {"иин": "900101100005"}, ["иин"]),
    ("ИИН из 9 цифр", {"иин": "900101323"}, ["иин"]),
    ("Опечатка в трек-номере (контрольная цифра)", {"трек_номер": "RR123456786ES"}, ["трек_номер"]),
    ("Трек-номер без букв в начале", {"трек_номер": "123456785ES"}, ["трек_номер"]),
    ("Индекс Астаны при городе Алматы", {"индекс": "010000"}, ["индекс"]),
    ("В телефоне 9 цифр после +7", {"телефон": "+7713123456"}, ["телефон"]),
    ("Городской номер вместо мобильного", {"телефон": "+7 713 123 45 67"}, ["телефон"]),
    ("Астана, новый формат индекса", {"город": "Астана", "индекс": "Z05H9E"}, []),
    ("Две ошибки: неверный ИИН и иностранный телефон",
     {"иин": "931982907123", "телефон": "+34 600 111 222"}, ["иин", "телефон"]),
]

CHECKED = ["трек_номер", "иин", "индекс", "телефон"]

BARE_PROMPT = """Ты проверяешь данные получателя посылки, которая едет из Испании в Казахстан.
Найди поля, в которых есть ошибка (неверный формат, опечатка, несоответствие).
Проверяй только поля: трек_номер, иин, индекс, телефон.
Верни ТОЛЬКО JSON: {"ошибки": [список названий полей с ошибкой]}. Если ошибок нет — пустой список.

Данные:
"""


def product_check(d: dict) -> list[str]:
    bad = []
    if check_track(d["трек_номер"])[1]:
        bad.append("трек_номер")
    if check_iin(d["иин"]):
        bad.append("иин")
    if check_postcode(d["индекс"], d["город"]):
        bad.append("индекс")
    if check_phone(d["телефон"]):
        bad.append("телефон")
    return bad


def bare_gemma_check(d: dict) -> list[str] | None:
    text = BARE_PROMPT + json.dumps(d, ensure_ascii=False, indent=2)
    for attempt in range(3):
        try:
            # тот же устойчивый вызов, что и в боте: повторы и запасные модели Gemma 4
            found = app.call_model(text).get("ошибки", [])
            if app.LAST_MODEL_USED not in MODELS_USED:
                MODELS_USED.append(app.LAST_MODEL_USED)
            return sorted(f for f in found if f in CHECKED)
        except Exception as e:
            print(f"  сбой API, повтор через 10 с: {str(e)[:80]}")
            time.sleep(10)
    return None


def main():
    rows, bare_ok, ours_ok, counted = [], 0, 0, 0
    for title, changes, expected in CASES:
        data = {**VALID, **changes}
        print(f"→ {title}")
        bare = bare_gemma_check(data)
        ours = sorted(product_check(data))
        exp = sorted(expected)
        b = bare == exp
        o = ours == exp
        if bare is not None:  # кейсы со сбоем API не учитываем ни для одной стороны
            counted += 1
            bare_ok += b
            ours_ok += o
        fmt = lambda x: ", ".join(FIELD_LABELS[f] for f in x) if x else "нет ошибок"
        rows.append(
            f"| {title} | {fmt(exp)} | {fmt(bare) if bare is not None else 'сбой API'} {'✅' if b else '❌'} "
            f"| {fmt(ours)} {'✅' if o else '❌'} |"
        )

    table = "\n".join(
        [
            f"# Результаты сравнения (модель: {', '.join(f'`{m}`' for m in MODELS_USED) or f'`{MODEL}`'})",
            "",
            "| Кейс | Ожидаемые ошибки | Голая Gemma 4 | Forwarder AI Operator |",
            "|---|---|---|---|",
            *rows,
            "",
            f"**Итого: голая Gemma 4 — {bare_ok} из {counted}, Forwarder AI Operator — {ours_ok} из {counted}.**"
            + (f" ({len(CASES) - counted} кейс(ов) пропущено из-за сбоя API.)" if counted < len(CASES) else ""),
        ]
    )
    print("\n" + table)
    with open("eval_results.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nСохранено в eval_results.md")


if __name__ == "__main__":
    main()
