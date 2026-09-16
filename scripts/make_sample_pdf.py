"""Создаёт docs/sample_agreement.pdf — маленький фиктивный договор для проверки.

Нужен, чтобы можно было прогнать весь конвейер сразу после клонирования, не ища
свои PDF. Пишем минимальный PDF руками, без сторонних библиотек.

Текст английский: встроенные шрифты PDF (Helvetica) не умеют кириллицу, а тащить
ради примера шрифтовый файл — лишнее. Для своих русских документов просто
положите их в docs/ — pypdf их прочитает нормально.

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

PAGE_WIDTH, PAGE_HEIGHT = 595, 842
MARGIN_LEFT, TOP, LINE_HEIGHT, FONT_SIZE = 50, 790, 16, 11

PAGES: list[list[str]] = [
    [
        "SERVICE AGREEMENT No. 42-2026",
        "",
        "1. SUBJECT OF THE AGREEMENT",
        "",
        "The Contractor undertakes to provide software maintenance services",
        "for the Customer's internal document management system, and the",
        "Customer undertakes to accept and pay for these services.",
        "",
        "2. WARRANTY",
        "",
        "The warranty period for all delivered components is twenty four (24)",
        "months from the date of the signed acceptance certificate. The warranty",
        "does not cover defects caused by unauthorised modification of the source",
        "code by the Customer or by third parties acting on the Customer's behalf.",
        "",
        "Warranty claims must be submitted in writing and must include the",
        "acceptance certificate number and a reproducible description of the defect.",
    ],
    [
        "3. PAYMENT TERMS",
        "",
        "The monthly service fee is 180,000 roubles, value added tax included.",
        "Invoices are issued on the first business day of each month and must be",
        "paid within ten (10) banking days from the date of receipt.",
        "",
        "Late payment incurs a penalty of 0.1 percent of the outstanding amount",
        "for each day of delay, capped at 10 percent of the monthly service fee.",
        "",
        "4. TERMINATION",
        "",
        "Either Party may terminate this Agreement unilaterally by giving thirty",
        "(30) calendar days prior written notice sent by registered mail to the",
        "address specified in Section 8.",
        "",
        "The Agreement may also be terminated immediately by mutual written",
        "consent of both Parties, in which case settlement of all outstanding",
        "invoices must be completed within five (5) banking days.",
    ],
    [
        "5. RESPONSE TIME AND SUPPORT",
        "",
        "The Contractor guarantees the following response times for incidents",
        "reported through the official support channel:",
        "",
        "  - Critical incidents (production outage): 2 hours",
        "  - High severity (major function unavailable): 8 business hours",
        "  - Normal severity (degraded behaviour): 3 business days",
        "",
        "Support is available on business days from 09:00 to 19:00 Moscow time.",
        "",
        "6. CONFIDENTIALITY",
        "",
        "Each Party shall keep confidential any technical and commercial",
        "information received from the other Party. This obligation survives",
        "termination of the Agreement for a period of three (3) years.",
    ],
]


def escape(text: str) -> bytes:
    for old, new in (("\\", r"\\"), ("(", r"\("), (")", r"\)")):
        text = text.replace(old, new)
    return text.encode("cp1252", errors="replace")


def content_stream(lines: list[str]) -> bytes:
    parts = [b"BT", f"/F1 {FONT_SIZE} Tf".encode(),
             f"{MARGIN_LEFT} {TOP} Td".encode(), f"{LINE_HEIGHT} TL".encode()]
    for line in lines:
        parts.append(b"(" + escape(line) + b") Tj T*")
    parts.append(b"ET")
    return b"\n".join(parts)


def build_pdf(pages: list[list[str]]) -> bytes:
    n = len(pages)
    # 1 Catalog, 2 Pages, 3 Font, затем на каждую страницу пара объектов.
    page_ids = [5 + 2 * i for i in range(n)]

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{pid} 0 R" for pid in page_ids), n
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
    ]

    for i, lines in enumerate(pages):
        stream = content_stream(lines)
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode()
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def main() -> None:
    target = Path(__file__).resolve().parent.parent / "docs" / "sample_agreement.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_pdf(PAGES))
    print(f"Готово: {target} ({len(PAGES)} стр.)")


if __name__ == "__main__":
    main()
