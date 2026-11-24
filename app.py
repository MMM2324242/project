import base64
import concurrent.futures
import io
import os
import re
import time
import unicodedata
from typing import List, Optional

import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from PIL import Image, ImageEnhance, ImageOps, ImageStat
from pydantic import BaseModel, Field

try:
    import pymupdf  # PyMuPDF
except ImportError:
    pymupdf = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None


# ---------------------------------------------------------
# 1. Data Schema (Pydantic) with Row Anchoring
# ---------------------------------------------------------
class TardinessRecord(BaseModel):
    row_index: int = Field(
        ...,
        description="1-based sequential row/line number from top to bottom to keep each student aligned.",
    )
    student_name: str = Field(
        ..., description="Student name in Arabic (translated if non-Arabic)."
    )
    delay_number: int = Field(
        ..., description="Numeric sequence/count of the tardiness incident for this row."
    )
    delay_duration: str = Field(
        ..., description="Exact time or duration of delay for this specific row."
    )
    reason: str = Field(
        ...,
        description=(
            "The exact, distinct excuse written on THIS SAME ROW after 'السبب' or 'Reason' "
            "(translated to Arabic if non-Arabic). Never copy or repeat a reason from another row "
            "unless the source explicitly writes the identical excuse on this row."
        ),
    )


class TardinessReport(BaseModel):
    records: List[TardinessRecord] = Field(
        default_factory=list,
        description="Ordered list of all tardiness records extracted strictly row by row.",
    )


# ---------------------------------------------------------
# 2. Text & Image Preprocessing Utilities
# ---------------------------------------------------------
TASHKEEL_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

AR_PATTERN = re.compile(
    r"اسم\s+الطالب(?:/ة|\s+أو\s+الطالبة|\s+ة)?\s*[:\-]?\s*(?P<name>.+?)\s*[|\-,]?\s*"
    r"(?:ال)?ت[اأإآ]خير\s+رقم\s*[:\-]?\s*(?P<delay_no>\d+)\s*[|\-,]?\s*"
    r"(?:كم\s+)?مد[ةه]\s+(?:ال)?ت[اأإآ]خير\s*[:\-]?\s*(?P<duration>[^\s|]+(?:\s+دقيق[ةه])?)\s*[|\-,]?\s*"
    r"السبب\s*[:\-]?\s*(?P<reason>.+?)(?=\s*(?:اسم\s+الطالب(?:/ة|\s+أو\s+الطالبة|\s+ة)?|$))",
    re.UNICODE,
)

EN_PATTERN = re.compile(
    r"(?:Student\s+Name|Name)\s*[:\-]\s*(?P<name>.+?)\s*[|,;]\s*"
    r"(?:Delay|Tardiness)\s*(?:#|No\.?|Number)?\s*[:\-]?\s*(?P<delay_no>\d+)\s*[|,;]\s*"
    r"(?:Duration|Time)\s*[:\-]\s*(?P<duration>[^|,;]+?)\s*[|,;]\s*"
    r"Reason\s*[:\-]\s*(?P<reason>.+?)(?=\s*(?:Student\s+Name|Name\s*[:\-]|$))",
    re.IGNORECASE,
)


def normalize_arabic_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return TASHKEEL_RE.sub("", text).strip()


def prepare_image_bytes(raw_bytes: bytes, for_ocr: bool = False, max_side: int = 1400) -> bytes:
    """
    Fast preprocessing: fixes EXIF orientation, inverts dark-mode images,
    upscales only when tiny, and returns lightweight JPEG bytes.
    """
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img).convert("L")

    # Invert dark-mode screenshots (white text on dark background)
    if ImageStat.Stat(img).mean[0] < 128:
        img = ImageOps.invert(img)

    if for_ocr:
        if min(img.size) < 1000:
            img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.6)
    else:
        # Keep Vision payloads fast while ensuring tiny screenshots stay readable
        if min(img.size) < 550:
            img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.25)

    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def decode_text_bytes(raw_bytes: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def pdf_to_content_blocks(pdf_bytes: bytes, for_ocr: bool = False) -> tuple[str, list[bytes]]:
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    extracted_texts = []
    scanned_page_images = []

    for page in doc:
        txt = normalize_arabic_text(page.get_text("text") or "")
        if len(txt) >= 15:
            extracted_texts.append(txt)
        else:
            pix = page.get_pixmap(dpi=170)
            scanned_page_images.append(prepare_image_bytes(pix.tobytes("png"), for_ocr=for_ocr))

    return "\n".join(extracted_texts).strip(), scanned_page_images


# ---------------------------------------------------------
# 3. Extraction Engines (Fast Regex + Gemini + Ollama)
# ---------------------------------------------------------
EXTRACTION_PROMPT = """
Extract every student tardiness record strictly line by line from top to bottom.
Important accuracy rules:
- Treat each line/row independently.
- Pair each student strictly with the exact delay number, delay duration, and reason written on THAT SAME horizontal line.
- NEVER copy, carry over, or repeat the reason from one student to another unless the source genuinely shows the exact same words on both lines.
- If any student name or reason is in English or another language, translate it accurately into natural Arabic.
- Keep delay durations in standard digits (0-9).
""".strip()


def batch_translate_to_arabic(texts: list[str]) -> list[str]:
    """Translate only non-Arabic items in one combined network request for speed."""
    if not texts or GoogleTranslator is None:
        return texts

    indices_to_translate = [i for i, t in enumerate(texts) if re.search(r"[A-Za-z]", t)]
    if not indices_to_translate:
        return texts

    translator = GoogleTranslator(source="auto", target="ar")
    to_translate = [texts[i] for i in indices_to_translate]
    joined = "\n".join(t.replace("\n", " ") for t in to_translate)

    if len(joined) <= 4500:
        try:
            translated_lines = translator.translate(joined).splitlines()
            if len(translated_lines) == len(to_translate):
                result = list(texts)
                for idx, val in zip(indices_to_translate, translated_lines):
                    result[idx] = val.strip()
                return result
        except Exception:
            pass

    # Safe fallback per unique string (avoids repeating API calls for identical words)
    cache: dict[str, str] = {}
    result = list(texts)
    for idx in indices_to_translate:
        src = texts[idx]
        if src not in cache:
            try:
                cache[src] = translator.translate(src[:4500])
            except Exception:
                cache[src] = src
        result[idx] = cache[src]
    return result


def extract_regex_from_text(text_content: str) -> list[TardinessRecord]:
    """Fast local regex extractor supporting both separated and merged lines."""
    cleaned = normalize_arabic_text(text_content)
    raw_items: list[tuple[str, int, str, str]] = []

    for raw_line in cleaned.splitlines():
        line = raw_line.strip().strip('"').strip("'")
        if not line:
            continue

        ar_matches = list(AR_PATTERN.finditer(line))
        if ar_matches:
            for m in ar_matches:
                raw_items.append(
                    (
                        m.group("name").strip(),
                        int(m.group("delay_no")),
                        m.group("duration").strip(),
                        m.group("reason").strip(),
                    )
                )
            continue

        en_matches = list(EN_PATTERN.finditer(line))
        if en_matches:
            for m in en_matches:
                raw_items.append(
                    (
                        m.group("name").strip(),
                        int(m.group("delay_no")),
                        m.group("duration").strip(),
                        m.group("reason").strip(),
                    )
                )

    if not raw_items:
        return []

    names = batch_translate_to_arabic([x[0] for x in raw_items])
    durations = batch_translate_to_arabic([x[2] for x in raw_items])
    reasons = batch_translate_to_arabic([x[3] for x in raw_items])

    return [
        TardinessRecord(
            row_index=i + 1,
            student_name=names[i],
            delay_number=raw_items[i][1],
            delay_duration=durations[i],
            reason=reasons[i],
        )
        for i in range(len(raw_items))
    ]


def extract_with_gemini(
    client,
    model_name: str,
    thinking_level: str,
    text_content: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
) -> list[TardinessRecord]:
    from google.genai import types

    parts = [EXTRACTION_PROMPT]
    if image_bytes is not None:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    if text_content:
        parts.append(f"Document Text:\n{text_content}")

    candidate_models = []
    for m in [model_name.strip(), "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    last_exc = None
    for current_model in candidate_models:
        if "2.5" in current_model:
            config = types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=TardinessReport,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                response_mime_type="application/json",
                response_schema=TardinessReport,
            )

        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=parts,
                    config=config,
                )
                parsed = TardinessReport.model_validate_json(response.text)
                return parsed.records
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

    raise last_exc


def extract_with_ollama(
    model_name: str,
    text_content: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
) -> list[TardinessRecord]:
    from ollama import chat

    msg = {"role": "user", "content": EXTRACTION_PROMPT}
    if text_content:
        msg["content"] += f"\n\nDocument Text:\n{text_content}"
    if image_bytes is not None:
        msg["images"] = [base64.b64encode(image_bytes).decode("utf-8")]

    response = chat(
        model=model_name,
        messages=[msg],
        format=TardinessReport.model_json_schema(),
        options={
            "temperature": 0,
            "repeat_penalty": 1.15,
            "num_predict": 2048,
        },
    )
    parsed = TardinessReport.model_validate_json(response.message.content)
    return parsed.records


def extract_offline_ocr(image_bytes: bytes) -> list[TardinessRecord]:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract")
    img = Image.open(io.BytesIO(image_bytes))
    ocr_text = pytesseract.image_to_string(img, lang="ara+eng", config="--psm 6")
    return extract_regex_from_text(ocr_text)


# ---------------------------------------------------------
# 4. RTL Excel Builder (4 Columns Only, AutoFilter Enabled)
# ---------------------------------------------------------
def build_rtl_excel(df: pd.DataFrame) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "سجل التأخيرات"
    ws.views.sheetView[0].rightToLeft = True
    ws.freeze_panes = "A2"

    headers = list(df.columns)
    ws.append(headers)
    for row in df.itertuples(index=False, name=None):
        ws.append(list(row))

    header_font = Font(name="Calibri", size=12, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    body_font = Font(name="Calibri", size=11, color="111111")
    thin_side = Side(style="thin", color="D9D9D9")
    cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    rtl_center_align = Alignment(
        horizontal="center", vertical="center", readingOrder=2, wrap_text=True
    )
    rtl_right_align = Alignment(
        horizontal="right", vertical="center", readingOrder=2, wrap_text=True
    )

    max_row = len(df) + 1
    max_col = len(headers)

    ws.row_dimensions[1].height = 28
    for col_idx in range(1, max_col + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = rtl_center_align
        cell.border = cell_border

    for r_idx in range(2, max_row + 1):
        ws.row_dimensions[r_idx].height = 24
        for c_idx in range(1, max_col + 1):
            cell = ws.cell(row=r_idx, column=c_idx)
            cell.font = body_font
            cell.border = cell_border
            cell.alignment = rtl_center_align if c_idx in (2, 3) else rtl_right_align

    for col_idx, col_name in enumerate(headers, start=1):
        col_letter = get_column_letter(col_idx)
        lengths = [len(str(col_name))] + [
            len(str(val)) for val in df.iloc[:, col_idx - 1].tolist()
        ]
        ws.column_dimensions[col_letter].width = min(max(max(lengths) * 1.35 + 6, 18), 55)

    if max_row >= 2:
        table_ref = f"A1:{get_column_letter(max_col)}{max_row}"
        tab = Table(displayName="TardinessTable", ref=table_ref)
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium9",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        ws.add_table(tab)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()


# ---------------------------------------------------------
# 5. Streamlit Application (English Interface)
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Auto Tardiness to Arabic Excel",
        page_icon="📊",
        layout="wide",
    )

    st.title("Auto Tardiness Extractor & Arabic Excel Formatter")
    st.markdown(
        "Upload any **PDF**, **TXT**, or **Image (PNG/JPG/WEBP)**. "
        "The program extracts each student's distinct reason, translates to **Arabic**, "
        "and generates a **Right-to-Left (RTL)** Excel file with **AutoFilter**."
    )

    with st.sidebar:
        st.header("Extraction Settings")
        mode = st.radio(
            "Select Processing Engine",
            options=[
                "Gemini AI (Fastest Cloud Vision & Text)",
                "Ollama Vision (100% Local AI)",
                "Offline Classic (Regex + Tesseract OCR + Google Translate)",
            ],
        )

        fast_regex_text = st.checkbox(
            "Instant local Regex for structured TXT/PDF",
            value=True,
            help="Skips AI API calls when text lines already match standard Arabic/English patterns.",
        )

        gemini_key = ""
        gemini_model = "gemini-3.8-flash"
        thinking_level = "low"
        ollama_model = "gemma4"

        if mode.startswith("Gemini"):
            gemini_key = st.text_input(
                "Gemini API Key",
                value=os.getenv("GEMINI_API_KEY", ""),
                type="password",
            )
            gemini_model = st.selectbox(
                "Gemini Model",
                options=[
                    "gemini-3.8-flash",
                    "gemini-3.5-flash-lite",
                    "gemini-2.5-flash",
                ],
                index=0,
                help="If the selected model is busy (503), the app automatically retries and falls back to the others.",
            )
            thinking_level = st.selectbox(
                "Speed vs. Accuracy (Gemini 3+)",
                options=["minimal", "low", "medium"],
                index=1,
                help="'minimal' is fastest; 'low' improves row-by-row accuracy on dense images.",
            )
        elif mode.startswith("Ollama"):
            ollama_model = st.text_input("Local Ollama Model", value="gemma4")

    uploaded_files = st.file_uploader(
        "Drag and drop your files here (PDF, TXT, PNG, JPG, JPEG, WEBP)",
        type=["pdf", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Waiting for file upload. No manual file paths needed.")
        return

    if st.button("Process Files & Generate Excel", type="primary"):
        if mode.startswith("Gemini") and not gemini_key.strip():
            st.error("Please enter your Gemini API Key in the sidebar.")
            return

        gemini_client = None
        if mode.startswith("Gemini"):
            from google import genai
            gemini_client = genai.Client(api_key=gemini_key)

        is_ocr_mode = mode.startswith("Offline")

        def process_single_file(file_idx: int, name: str, file_bytes: bytes) -> tuple[int, str, list[TardinessRecord]]:
            ext = os.path.splitext(name)[1].lower()
            text_data = None
            images_to_process: list[bytes] = []

            if ext == ".txt":
                text_data = normalize_arabic_text(decode_text_bytes(file_bytes))
            elif ext == ".pdf":
                text_data, images_to_process = pdf_to_content_blocks(file_bytes, for_ocr=is_ocr_mode)
            else:
                images_to_process = [prepare_image_bytes(file_bytes, for_ocr=is_ocr_mode)]

            extracted: list[TardinessRecord] = []

            if text_data:
                regex_hits = extract_regex_from_text(text_data) if fast_regex_text else []
                if regex_hits:
                    extracted.extend(regex_hits)
                elif mode.startswith("Gemini"):
                    extracted.extend(
                        extract_with_gemini(
                            gemini_client, gemini_model, thinking_level, text_content=text_data
                        )
                    )
                elif mode.startswith("Ollama"):
                    extracted.extend(extract_with_ollama(ollama_model, text_content=text_data))
                else:
                    extracted.extend(extract_regex_from_text(text_data))

            for img_b in images_to_process:
                if mode.startswith("Gemini"):
                    extracted.extend(
                        extract_with_gemini(
                            gemini_client, gemini_model, thinking_level, image_bytes=img_b
                        )
                    )
                elif mode.startswith("Ollama"):
                    extracted.extend(extract_with_ollama(ollama_model, image_bytes=img_b))
                else:
                    extracted.extend(extract_offline_ocr(img_b))

            return file_idx, name, extracted

        results_by_index: dict[int, list[TardinessRecord]] = {}
        progress = st.progress(0)

        with st.spinner("Processing files..."):
            # Run cloud/offline tasks concurrently; keep Ollama sequential to avoid local GPU/CPU contention
            max_workers = 1 if mode.startswith("Ollama") else min(4, len(uploaded_files))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_single_file, idx, f.name, f.getvalue())
                    for idx, f in enumerate(uploaded_files)
                ]
                done_count = 0
                for future in concurrent.futures.as_completed(futures):
                    done_count += 1
                    try:
                        f_idx, _, recs = future.result()
                        results_by_index[f_idx] = recs
                    except Exception as exc:
                        st.error(f"Error processing file: {exc}")
                    progress.progress(done_count / len(uploaded_files))

        all_rows = []
        for idx in sorted(results_by_index.keys()):
            for rec in results_by_index[idx]:
                all_rows.append(
                    {
                        "اسم الطالب/ة": rec.student_name,
                        "رقم التأخير": rec.delay_number,
                        "مدة التأخير": rec.delay_duration,
                        "السبب": rec.reason,
                    }
                )

        if not all_rows:
            st.warning("No tardiness records could be extracted. Try switching to Gemini AI or check image clarity.")
            return

        df_result = pd.DataFrame(all_rows)
        st.success(f"Extracted {len(df_result)} tardiness record(s) in Arabic!")
        st.dataframe(df_result, use_container_width=True)

        excel_bytes = build_rtl_excel(df_result)
        st.download_button(
            label="Download Arabic RTL Excel (.xlsx)",
            data=excel_bytes,
            file_name="tardiness_report_rtl.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()