import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
import httpx
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image to CSV")

VLLM_URL = os.getenv("VLLM_URL", "http://vllm-ocr:8000")
LLM_URL = os.getenv("LLM_URL", "http://vllm-llm:8000")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

# Minimum long-edge resolution before upscaling; GLM-OCR accuracy improves
# significantly when text is at least ~1500px on the long edge.
_MIN_LONG_EDGE = 1500


def _preprocess_image(raw_bytes: bytes, content_type: str) -> tuple[bytes, str]:
    """Upscale small images and return (png_bytes, 'image/png').

    GLM-OCR's CogViT encoder benefits from higher resolution input.
    Images whose long edge is already >= _MIN_LONG_EDGE are returned as-is
    (re-encoded as PNG for lossless quality).
    """
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size
    long_edge = max(w, h)
    if long_edge < _MIN_LONG_EDGE:
        scale = _MIN_LONG_EDGE / long_edge
        new_w, new_h = int(w * scale), int(h * scale)
        # LANCZOS gives best quality for document upscaling
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Mild unsharp mask improves OCR on blurry scans after upscaling
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue(), "image/png"


# ── HTML table parser ──────────────────────────────────────────────────────
class _HTMLTableParser(HTMLParser):
    """Extract rows from an HTML <table>, tracking thead/tbody and span attributes."""

    def __init__(self):
        super().__init__()
        self.thead: list[list[tuple[str, int, int]]] = []  # rows of (text, rowspan, colspan)
        self.tbody: list[list[tuple[str, int, int]]] = []
        self._section = "tbody"
        self._cur_row: list[tuple[str, int, int]] | None = None
        self._cur_cell: list[str] | None = None
        self._in_cell = False
        self._cur_rs = 1
        self._cur_cs = 1

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "thead":
            self._section = "thead"
        elif tag == "tbody":
            self._section = "tbody"
        elif tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = []
            self._in_cell = True
            try:
                self._cur_rs = int(d.get("rowspan", 1))
            except (ValueError, TypeError):
                self._cur_rs = 1
            try:
                self._cur_cs = int(d.get("colspan", 1))
            except (ValueError, TypeError):
                self._cur_cs = 1

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cur_row is not None:
            text = "".join(self._cur_cell or []).strip()
            self._cur_row.append((text, self._cur_rs, self._cur_cs))
            self._cur_cell = None
            self._in_cell = False
        elif tag == "tr" and self._cur_row is not None:
            dest = self.thead if self._section == "thead" else self.tbody
            dest.append(self._cur_row)
            self._cur_row = None

    def handle_data(self, data):
        if self._in_cell and self._cur_cell is not None:
            self._cur_cell.append(data)


def _expand_rows(raw_rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    """Expand (text, rowspan, colspan) rows into a flat 2D list, honouring span attributes."""
    pending: dict[int, tuple[int, str]] = {}  # col -> (remaining_rows, text)
    result: list[list[str]] = []

    for raw_row in raw_rows:
        row_dict: dict[int, str] = {}
        col = 0
        for text, rs, cs in raw_row:
            # Drain pending rowspan cells that occupy columns at or before current col
            while col in pending:
                rem, val = pending[col]
                row_dict[col] = val
                if rem > 1:
                    pending[col] = (rem - 1, val)
                else:
                    del pending[col]
                col += 1
            # Fill colspan columns with this cell's text
            for c in range(cs):
                row_dict[col + c] = text
                if rs > 1:
                    pending[col + c] = (rs - 1, text)
            col += cs
        # Drain any trailing pending rowspan columns not reached by actual cells
        for pc in sorted(k for k in pending if k not in row_dict):
            rem, val = pending[pc]
            row_dict[pc] = val
            if rem > 1:
                pending[pc] = (rem - 1, val)
            else:
                del pending[pc]
        if row_dict:
            max_col = max(row_dict.keys()) + 1
            result.append([row_dict.get(i, "") for i in range(max_col)])

    return result


def _combine_header_rows(head_rows: list[list[str]]) -> list[str]:
    """Combine multi-row headers into single column names.

    Rowspan repeats are deduplicated; colspan groups are combined with their children.
    e.g. row0=["MORNING","MORNING"] + row1=["IN","OUT"] → ["MORNING IN","MORNING OUT"]
    """
    if not head_rows:
        return []
    if len(head_rows) == 1:
        return head_rows[0]
    num_cols = max(len(r) for r in head_rows)
    headers = []
    for col in range(num_cols):
        parts: list[str] = []
        seen: set[str] = set()
        for row in head_rows:
            val = row[col] if col < len(row) else ""
            if val and val not in seen:
                parts.append(val)
                seen.add(val)
        headers.append(" ".join(parts) if parts else f"Column {col + 1}")
    return headers


def _extract_non_table_text(raw: str) -> dict:
    """Split raw OCR output into text_above, table_html, text_below."""
    result = {"text_above": [], "text_below": []}

    # Find the <table...>...</table> block
    match = re.search(r"<table[\s>].*?</table>", raw, re.DOTALL | re.IGNORECASE)
    if not match:
        # Try markdown table
        lines = raw.strip().split("\n")
        above, below = [], []
        in_table = False
        past_table = False
        for line in lines:
            stripped = line.strip()
            if "|" in stripped and not past_table:
                in_table = True
            elif in_table and "|" not in stripped:
                in_table = False
                past_table = True
            if not in_table and not past_table and stripped:
                above.append(stripped)
            elif past_table and stripped:
                below.append(stripped)
        result["text_above"] = above
        result["text_below"] = below
        return result

    before = raw[: match.start()].strip()
    after = raw[match.end() :].strip()

    if before:
        result["text_above"] = [
            line.strip()
            for line in before.split("\n")
            if line.strip()
        ]
    if after:
        result["text_below"] = [
            line.strip()
            for line in after.split("\n")
            if line.strip()
        ]
    return result


def parse_html_table(html: str) -> dict | None:
    """Parse an HTML table string into headers + rows, handling colspan/rowspan correctly."""
    if "<table" not in html.lower():
        return None

    parser = _HTMLTableParser()
    parser.feed(html)

    # Expand thead rows into flat grid
    head_rows = _expand_rows(parser.thead)
    if head_rows:
        headers = _combine_header_rows(head_rows)
        data_rows = _expand_rows(parser.tbody)
    else:
        # No explicit thead — first tbody row is the header
        all_rows = _expand_rows(parser.tbody)
        if not all_rows:
            return None
        headers = all_rows[0]
        data_rows = all_rows[1:]

    if not headers:
        return None

    # Normalise column count
    n = max(len(headers), *(len(r) for r in data_rows) if data_rows else [0])
    while len(headers) < n:
        headers.append(f"Column {len(headers) + 1}")

    padded_rows = []
    for row in data_rows:
        if not any(c != "" for c in row):
            continue  # skip fully empty rows
        padded = (row + [""] * n)[:n]
        padded_rows.append(padded)

    return {"headers": headers, "rows": padded_rows}


def parse_markdown_table(md_text: str) -> dict:
    """Parse a markdown pipe table into headers + rows."""
    lines = [line.strip() for line in md_text.strip().split("\n") if line.strip()]

    table_lines = [line for line in lines if "|" in line]
    if not table_lines:
        return {"headers": ["Content"], "rows": [[line] for line in lines if line]}

    parsed = []
    for line in table_lines:
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        parsed.append(cells)

    if not parsed:
        return {"headers": ["Content"], "rows": [[md_text.strip()]]}

    headers = parsed[0]
    rows = []
    for row in parsed[1:]:
        if all(re.match(r"^[-:\s]+$", cell) for cell in row):
            continue
        while len(row) < len(headers):
            row.append("")
        rows.append(row[: len(headers)])

    return {"headers": headers, "rows": rows}


def parse_table(text: str) -> dict:
    """Try HTML first, then markdown, then plain-text fallback."""
    result = parse_html_table(text)
    if result and result["headers"]:
        return result
    return parse_markdown_table(text)


def _parse_csv_template(csv_text: str) -> dict:
    """Parse a structured CSV template into header_fields, table_headers, footer_fields.

    Template format:
      - 2-column rows with col[0] ending in ':' → metadata fields
      - First row with 3+ columns → table header row
      - Metadata rows before the table header → header_fields
      - Metadata rows after the table header → footer_fields
    """
    reader = csv.reader(io.StringIO(csv_text))
    all_rows = [row for row in reader]

    header_fields: list[list[str]] = []  # [[label, value], ...]
    table_headers: list[str] = []
    footer_fields: list[list[str]] = []
    found_table = False

    for row in all_rows:
        stripped = [c.strip() for c in row]
        non_empty = [c for c in stripped if c]

        # Skip completely empty rows
        if not non_empty:
            continue

        # If we haven't found the table header yet, look for it
        if not found_table:
            # Row with 3+ non-empty columns is the table header
            if len(stripped) >= 3 and len(non_empty) >= 3:
                table_headers = stripped
                found_table = True
            else:
                # 2-column metadata row
                label = stripped[0] if len(stripped) > 0 else ""
                value = stripped[1] if len(stripped) > 1 else ""
                if label:
                    header_fields.append([label, value])
        else:
            # After the table header: check if it's a data row (mostly empty) or footer metadata
            if len(stripped) <= 2 and any(c for c in stripped if c):
                label = stripped[0] if len(stripped) > 0 else ""
                value = stripped[1] if len(stripped) > 1 else ""
                if label:
                    footer_fields.append([label, value])
            # Skip template data rows (all empty or matching table column count)

    return {
        "header_fields": header_fields,
        "table_headers": table_headers,
        "footer_fields": footer_fields,
    }


@app.post("/api/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    csv_template: UploadFile | None = File(None),
):
    """Upload an image and extract table data using GLM-OCR."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 20 MB)")

    # Preprocess: upscale small images and normalise to PNG for lossless quality
    contents, mime_type = _preprocess_image(contents, file.content_type or "image/png")
    b64_image = base64.b64encode(contents).decode("utf-8")

    # Parse CSV template if provided
    template = {"header_fields": [], "table_headers": [], "footer_fields": []}
    if csv_template and csv_template.filename:
        csv_bytes = await csv_template.read()
        try:
            csv_text = csv_bytes.decode("utf-8-sig")
            template = _parse_csv_template(csv_text)
        except Exception:
            pass

    # Determine what's needed before building the prompt
    need_fields = bool(template["header_fields"] or template["footer_fields"])
    has_user_prompt = bool(prompt.strip())
    all_meta_fields = template.get("header_fields", []) + template.get("footer_fields", [])
    field_labels = [row[0].rstrip(":").strip() for row in all_meta_fields if row[0]]

    # ── Single comprehensive VLM call: OCR + extraction/transformation ──────
    instr_parts = [
        "Analyze this document image carefully.",
        "Return a single valid JSON object with the following keys:",
        "",
        '"full_text": Full transcription of all text in the document, preserving structure and layout.',
        '"headers": Array of column header strings for the primary table.',
        '"rows": Array of row arrays — each row is an array of cell strings. Do NOT include the header row.',
    ]
    if template["table_headers"]:
        instr_parts.append(
            f'  Use exactly these column headers for "headers": {json.dumps(template["table_headers"])}'
        )
    if need_fields:
        instr_parts.append('"fields": Extract these field values from the document header/footer:')
        for label in field_labels:
            instr_parts.append(f'  "{label}": ""')
    if has_user_prompt:
        instr_parts += ["", 'Transformation instructions — apply when populating "rows":', prompt.strip()]
    instr_parts += ["", "Output ONLY valid JSON. No markdown fences, no explanation."]

    payload = {
        "model": "qwen3.5-122b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                    {"type": "text", "text": "\n".join(instr_parts)},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            raw_result = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"OCR model error: {e.response.text}")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise HTTPException(503, "Cannot connect to OCR model server")

    raw_content = raw_result["choices"][0]["message"]["content"]
    raw_content = re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()
    logger.info("VLM response (first 2000): %s", raw_content[:2000])

    # Initialise table_data with safe defaults
    table_data: dict = {
        "headers": [], "rows": [], "text_above": [], "text_below": [],
        "raw": raw_content, "llm_applied": False, "llm_raw": raw_content,
    }
    full_ocr_text = raw_content  # fallback if JSON parse fails

    try:
        json_match = re.search(r"\{[\s\S]*\}", raw_content)
        if not json_match:
            raise ValueError("No JSON object found in VLM response")
        result_json = json.loads(json_match.group(0))

        full_ocr_text = result_json.get("full_text", raw_content)

        # Headers
        raw_headers = result_json.get("headers", [])
        if isinstance(raw_headers, list):
            table_data["headers"] = [str(h) for h in raw_headers]
        if template["table_headers"]:
            table_data["headers"] = template["table_headers"]

        # Rows — normalise to n columns, drop empty rows
        n = len(table_data["headers"])
        raw_rows = result_json.get("rows", [])
        if isinstance(raw_rows, list):
            cleaned: list[list[str]] = []
            for row in raw_rows:
                if isinstance(row, list):
                    r = [str(c) for c in row]
                    while len(r) < n:
                        r.append("")
                    if n:
                        r = r[:n]
                    if any(c for c in r):
                        cleaned.append(r)
            table_data["rows"] = cleaned

        if (need_fields or has_user_prompt) and table_data["rows"]:
            table_data["llm_applied"] = True

        # Fields extracted by the model
        if need_fields and "fields" in result_json and isinstance(result_json["fields"], dict):
            extracted_lower = {
                k.lower().strip().rstrip(":"): v for k, v in result_json["fields"].items()
            }

            def _fill(fields: list) -> list:
                out = []
                for row in fields:
                    label = row[0].rstrip(":").strip()
                    default = row[1] if len(row) > 1 else ""
                    value = extracted_lower.get(label.lower(), default)
                    out.append([label, value])
                return out

            if template["header_fields"]:
                table_data["template_header_fields"] = _fill(template["header_fields"])
            if template["footer_fields"]:
                table_data["template_footer_fields"] = _fill(template["footer_fields"])
            table_data["llm_applied"] = True

    except Exception as e:
        logger.warning("VLM JSON parse failed (%s): %s — falling back to table parser", type(e).__name__, e)
        fallback = parse_table(raw_content)
        table_data["headers"] = fallback.get("headers", [])
        table_data["rows"] = fallback.get("rows", [])
        non_table = _extract_non_table_text(raw_content)
        table_data["text_above"] = non_table["text_above"]
        table_data["text_below"] = non_table["text_below"]

    # Ensure template fields exist in response even if extraction failed
    if template["header_fields"] and "template_header_fields" not in table_data:
        table_data["template_header_fields"] = template["header_fields"]
    if template["footer_fields"] and "template_footer_fields" not in table_data:
        table_data["template_footer_fields"] = template["footer_fields"]

    # Split full OCR text into header lines (before table) and footer lines (after table).
    # Uses the extracted grid data to locate the table region in the text, then keeps only
    # lines that look like metadata key:value fields.
    _time_re = re.compile(r'\b\d{1,2}:\d{2}\b')

    def _is_field_line(ln: str) -> bool:
        ln = ln.strip()
        if not ln:
            return False
        idx = ln.find(':')
        if not (0 < idx <= 60):
            return False
        label = ln[:idx].strip()
        value = ln[idx + 1:].strip()
        if not value:
            return False
        if not any(c.isalpha() for c in label):
            return False
        # A line with 2+ time patterns is a table data row, not a field
        if len(_time_re.findall(ln)) >= 2:
            return False
        return True

    # Build set of grid values for cross-referencing
    grid_vals: set[str] = set()
    for row in table_data.get("rows", []):
        for cell in row:
            v = str(cell).strip().lower()
            if v and len(v) > 1:
                grid_vals.add(v)
    for h in table_data.get("headers", []):
        v = h.strip().lower()
        if v and len(v) > 1:
            grid_vals.add(v)

    def _is_table_line(ln: str) -> bool:
        """True if this line is likely a table row (dense grid values or 2+ time patterns)."""
        if len(_time_re.findall(ln)) >= 2:
            return True
        tokens = re.findall(r'[\w:×·]+', ln.lower())
        if not tokens:
            return False
        hits = sum(1 for t in tokens if t in grid_vals)
        return hits / len(tokens) >= 0.4

    ocr_lines_all = full_ocr_text.splitlines()

    # Find the table region in the OCR text
    table_start: int | None = None
    table_end: int | None = None
    for i, ln in enumerate(ocr_lines_all):
        if _is_table_line(ln):
            if table_start is None:
                table_start = i
            table_end = i

    # Collect header and footer field lines based on position relative to table
    header_ocr_lines: list[str] = []
    footer_ocr_lines: list[str] = []
    for i, ln in enumerate(ocr_lines_all):
        if not _is_field_line(ln):
            continue
        # Cross-reference: skip if the value after the colon is just a raw grid cell
        val_part = ln[ln.find(':') + 1:].strip().lower()
        if val_part in grid_vals:
            continue
        if table_start is None or i < table_start:
            header_ocr_lines.append(ln.strip())
        elif table_end is not None and i > table_end:
            footer_ocr_lines.append(ln.strip())

    table_data["ocr_text_lines"] = header_ocr_lines
    table_data["ocr_footer_lines"] = footer_ocr_lines
    return table_data


@app.get("/api/health")
async def health():
    """Health check – pings both vLLM OCR and LLM."""
    async def _check(url):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{url}/health")
                return r.status_code == 200
        except Exception:
            return False
    ocr_ok, llm_ok = await asyncio.gather(_check(VLLM_URL), _check(LLM_URL))
    return {"status": "ok", "ocr_ready": ocr_ok, "llm_ready": llm_ok, "model_ready": ocr_ok}


# Serve the frontend
app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
