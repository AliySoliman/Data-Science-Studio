import os
import re
import unicodedata
import requests
import tempfile
from fpdf import FPDF
from fpdf.enums import XPos, YPos

_EMOJI_REPLACEMENTS = {
    "1️⃣": "1.", "2️⃣": "2.", "3️⃣": "3.", "4️⃣": "4.", "5️⃣": "5.",
    "✅": "PASS", "⚠️": "WARNING", "❌": "FAIL", "🔴": "FAIL", "🚨": "CRITICAL",
    "ℹ️": "INFO", "📅": "DATE", "📝": "NOTE", "🌲": "TREE",
    "📊": "", "🔍": "", "🎯": "", "📈": "", "🔗": "", "🤖": "",
    "🚀": "", "✍️": "", "🧠": "", "🏆": "",
    "💾": "", "📄": "", "📋": "", "🔖": "", "📂": "", "✨": "",
    "—": "-", "–": "-", "→": "->", "\u200d": "", "\ufe0f": "", "\u20e3": "",
}

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d\ufe0f\u20e3"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    for emoji, replacement in _EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, replacement)
    text = _EMOJI_PATTERN.sub("", text)

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"^#+\s*", "", text)

    return text.encode("latin-1", errors="ignore").decode("latin-1")


class ExecutivePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 18, 16)

        self.brand = (30, 58, 138)
        self.brand_dark = (15, 23, 42)
        self.brand_light = (219, 234, 254)
        self.gray_50 = (248, 250, 252)
        self.gray_100 = (241, 245, 249)
        self.gray_200 = (226, 232, 240)
        self.gray_500 = (100, 116, 139)
        self.gray_700 = (51, 65, 85)
        self.success = (22, 163, 74)
        self.warning = (217, 119, 6)
        self.danger = (220, 38, 38)

        self.report_title = "DSS AI Advisor"
        self.report_subtitle = "Strategy Report"

    def ensure_space(self, needed_height: float):
        if self.get_y() + needed_height > self.h - self.b_margin:
            self.add_page()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_draw_color(*self.gray_200)
        self.line(self.l_margin, 12, self.w - self.r_margin, 12)

        self.set_y(6)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.brand_dark)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(*self.gray_200)
        self.line(self.l_margin, self.get_y() - 2, self.w - self.r_margin, self.get_y() - 2)

        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.gray_500)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def cover_page(self, title=None, subtitle=None, meta_lines=None, cover_image_path=None):
        self.add_page()
        title = title or self.report_title
        subtitle = subtitle or self.report_subtitle

        if cover_image_path and os.path.exists(cover_image_path):
            self.image(cover_image_path, x=0, y=0, w=self.w, h=70)
        else:
            self.set_fill_color(*self.brand_dark)
            self.rect(0, 0, self.w, 90, style="F")

        self.set_xy(16, 20)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 24)
        self.multi_cell(0, 12, clean_text(title))

        self.set_font("Helvetica", "", 14)
        self.set_text_color(226, 232, 240)
        self.ln(2)
        self.multi_cell(0, 8, clean_text(subtitle))

        self.ln(18)
        self.set_text_color(*self.gray_700)

        if meta_lines:
            self.section_band("Report Snapshot")
            for line in meta_lines:
                self.bullet(line)

    def section_band(self, title):
        self.ensure_space(22)
        self.ln(4)
        self.set_fill_color(*self.brand)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, f"  {clean_text(title)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(3)
        self.set_text_color(*self.gray_700)

    def subsection(self, title):
        self.ensure_space(16)
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.brand_dark)
        self.cell(0, 7, clean_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*self.brand_light)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def _wrap_text_lines(self, text, width, font_family="Helvetica", font_style="", font_size=10.5):
        text = clean_text(text).strip()
        if not text:
            return []

        self.set_font(font_family, font_style, font_size)

        words = text.split()
        lines = []
        current = ""

        for word in words:
            test = word if not current else f"{current} {word}"
            if self.get_string_width(test) <= width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    def _estimate_multicell_height(self, text, width, line_height=5.8, font_family="Helvetica", font_style="", font_size=10.5):
        lines = self._wrap_text_lines(text, width, font_family, font_style, font_size)
        return max(line_height, len(lines) * line_height)

    def bullet(self, text, color=None, indent=0):
        cleaned = clean_text(text).strip()
        if not cleaned or cleaned == "-":
            return

        color = color or self.gray_700

        bullet_w = 5
        left_x = self.l_margin + indent
        text_w = self.w - self.r_margin - left_x - bullet_w

        needed_h = self._estimate_multicell_height(
            cleaned,
            width=text_w,
            line_height=5.8,
            font_family="Helvetica",
            font_style="",
            font_size=10.3,
        ) + 1.0

        self.ensure_space(needed_h + 1)

        start_y = self.get_y()
        self.set_xy(left_x, start_y)

        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*color)
        self.cell(bullet_w, 5.8, "-")

        self.set_xy(left_x + bullet_w, start_y)
        self.set_font("Helvetica", "", 10.3)
        self.set_text_color(*self.gray_700)
        self.multi_cell(text_w, 5.8, cleaned)
        self.ln(0.4)

    def paragraph(self, text, size=10.5):
        cleaned = clean_text(text).strip()
        if not cleaned or cleaned == "-":
            return

        needed_h = self._estimate_multicell_height(
            cleaned,
            width=self.w - self.l_margin - self.r_margin,
            line_height=6.0,
            font_family="Helvetica",
            font_style="",
            font_size=size,
        ) + 0.5

        self.ensure_space(needed_h + 1)

        self.set_font("Helvetica", "", size)
        self.set_text_color(*self.gray_700)
        self.multi_cell(0, 6.0, cleaned)
        self.ln(0.6)

    def _estimate_text_height(self, text: str, line_height: float = 5.5, chars_per_line: int = 85) -> float:
        cleaned = clean_text(text)
        lines = max(1, (len(cleaned) // chars_per_line) + 1)
        return lines * line_height

    def callout_box(self, title, body, tone="brand"):
        tones = {
            "brand": (self.brand, self.brand_light),
            "success": (self.success, (240, 253, 244)),
            "warning": (self.warning, (255, 251, 235)),
            "danger": (self.danger, (254, 242, 242)),
        }
        border, bg = tones[tone]

        x = self.l_margin
        y = self.get_y()
        w = self.w - self.l_margin - self.r_margin
        pad = 5

        title_h = self._estimate_text_height(title, line_height=5, chars_per_line=75)
        body_h = self._estimate_text_height(body, line_height=5.5, chars_per_line=90)
        total_h = 8 + title_h + body_h + pad + 6

        self.ensure_space(total_h + 5)
        y = self.get_y()

        self.set_fill_color(*bg)
        self.set_draw_color(*border)

        self.rect(x, y, w, 4, style="F")
        self.rect(x, y + 4, w, total_h, style="D")

        self.set_xy(x + pad, y + 7)
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*border)
        self.multi_cell(w - 2 * pad, 5, clean_text(title))

        self.set_x(x + pad)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.gray_700)
        self.multi_cell(w - 2 * pad, 5.5, clean_text(body))

        self.set_y(y + 4 + total_h + 4)
    
    def simple_table(self, headers, rows, col_widths=None):
        if not headers or not rows:
            return

        n_cols = len(headers)
        usable = self.w - self.l_margin - self.r_margin

        if not col_widths:
            col_widths = [usable / n_cols] * n_cols
        elif sum(col_widths) <= 10:
            factor = usable / sum(col_widths)
            col_widths = [w * factor for w in col_widths]

        self.ensure_space(20)

        self.set_fill_color(*self.brand)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9.5)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, clean_text(h), border=0, fill=True)
        self.ln()

        self.set_text_color(*self.gray_700)
        self.set_font("Helvetica", "", 9)

        for r_idx, row in enumerate(rows):
            fill = self.gray_50 if r_idx % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*fill)

            row_h = 7
            self.ensure_space(row_h + 2)

            for i, cell in enumerate(row):
                self.cell(col_widths[i], row_h, clean_text(str(cell))[:60], border=0, fill=True)
            self.ln()

        self.ln(3)

    def model_card(self, name, why, hyperparams, strength, tone="brand"):
        tones = {
            "brand": (self.brand, self.brand_light),
            "success": (self.success, (240, 253, 244)),
            "warning": (self.warning, (255, 251, 235)),
            "danger": (self.danger, (254, 242, 242)),
            "neutral": (self.gray_500, self.gray_100),
        }
        accent, bg = tones[tone]

        w = self.w - self.l_margin - self.r_margin
        pad = 5

        why = clean_text(why)
        hyperparams = clean_text(hyperparams)
        strength = clean_text(strength)
        name = clean_text(name)

        why_h = self._estimate_text_height(why, line_height=5.3, chars_per_line=78)
        hyper_h = self._estimate_text_height(hyperparams, line_height=5.3, chars_per_line=72)
        strength_h = self._estimate_text_height(strength, line_height=5.3, chars_per_line=80)

        h = 18 + why_h + hyper_h + strength_h + 16

        self.ensure_space(h + 6)
        x = self.l_margin
        y = self.get_y()

        self.set_fill_color(*bg)
        self.set_draw_color(*accent)
        self.rect(x, y, w, h, style="FD")

        self.set_fill_color(*accent)
        self.rect(x, y, w, 9, style="F")

        self.set_xy(x + pad, y + 2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 5, name)

        self.set_xy(x + pad, y + 12)
        self.set_text_color(*self.gray_700)

        self.set_font("Helvetica", "B", 9.5)
        self.cell(30, 5, "Why it fits:")
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(w - 2 * pad - 30, 5.3, why)

        self.set_x(x + pad)
        self.set_font("Helvetica", "B", 9.5)
        self.cell(38, 5, "Hyperparameters:")
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(w - 2 * pad - 38, 5.3, hyperparams)

        self.set_x(x + pad)
        self.set_font("Helvetica", "B", 9.5)
        self.cell(30, 5, "Strength:")
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(w - 2 * pad - 30, 5.3, strength)

        self.set_y(y + h + 4)

    def divider(self):
        self.ensure_space(8)
        self.ln(2)
        self.set_draw_color(*self.gray_200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)


def extract_model_cards(markdown_text: str):
    lines = normalize_markdown_lines(markdown_text)
    model_section = False
    table_lines = []

    for line in lines:
        if "Targeted Model Selection" in line:
            model_section = True
            continue
        if model_section:
            if line.startswith("## ") or line.startswith("# ") or "Evaluation & Validation Framework" in line:
                break
            if line.strip().startswith("|") and line.strip().endswith("|"):
                table_lines.append(line.strip())

    if len(table_lines) < 3:
        return []

    rows = []
    parsed = []
    for line in table_lines:
        parsed.append([clean_text(c.strip()) for c in line.split("|")[1:-1]])

    parsed = [r for r in parsed if not all(set(c.replace(" ", "")) <= {"-"} for c in r)]

    if len(parsed) < 2:
        return []

    for row in parsed[1:]:
        if len(row) >= 4:
            rows.append({
                "name": row[0],
                "why": row[1],
                "hyperparams": row[2],
                "strength": row[3],
            })
    return rows


def markdown_to_luxury_pdf(markdown_text: str, cover_image_path: str | None = None) -> bytes:
    pdf = ExecutivePDF()

    temp_cover_file = None
    original_cover_path = cover_image_path

    # Support remote cover images (GitHub raw URLs, etc.)
    if cover_image_path and (cover_image_path.startswith("http://") or cover_image_path.startswith("https://")):
        try:
            resp = requests.get(cover_image_path, timeout=10)
            if resp.status_code == 200:
                # Use a temp file to store the downloaded image
                suffix = ".png" if ".png" in cover_image_path.lower() else ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(resp.content)
                    temp_cover_file = tmp.name
                    cover_image_path = temp_cover_file
            else:
                print(f"DEBUG: Cover image download failed (HTTP {resp.status_code}): {cover_image_path}")
        except Exception as e:
            print(f"DEBUG: Cover image download error: {e}")

    pdf.cover_page(
        title="DSS AI Advisor",
        subtitle="Executive Strategy Report",
        meta_lines=[
            "Prepared as a structured modeling and data-readiness assessment.",
            "Designed for premium executive-style presentation.",
            "Optimized for clean Streamlit export as a polished PDF report.",
        ],
        cover_image_path=cover_image_path,
    )

    model_cards = extract_model_cards(markdown_text)

    lines = markdown_text.splitlines()
    table_buffer = []
    in_code = False
    in_model_section = False

    def flush_table():
        nonlocal table_buffer
        if not table_buffer:
            return

        cleaned_rows = []
        for idx, row in enumerate(table_buffer):
            if idx == 1 and all(set(cell.replace(" ", "")) <= {"-"} for cell in row):
                continue
            cleaned_rows.append(row)

        if len(cleaned_rows) >= 2:
            headers = cleaned_rows[0]
            rows = cleaned_rows[1:]
            pdf.simple_table(headers, rows)

        table_buffer = []

    prev_blank = False

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not stripped:
            flush_table()
            if not prev_blank:
                pdf.ln(1.2)
            prev_blank = True
            continue

        prev_blank = False

        if stripped == "-":
            continue

        if "Targeted Model Selection" in stripped:
            flush_table()
            in_model_section = True
            pdf.section_band("Targeted Model Selection")
            if model_cards:
                for idx, m in enumerate(model_cards):
                    tone = "brand" if idx == 0 else "neutral"
                    pdf.model_card(
                        m["name"],
                        m["why"],
                        m["hyperparams"],
                        m["strength"],
                        tone=tone,
                    )
            continue

        if in_model_section:
            if (
                "Evaluation & Validation Framework" in stripped
                or stripped.startswith("## 4.")
                or stripped.startswith("# 4.")
            ):
                in_model_section = False
            else:
                continue

        if stripped.startswith("```"):
            flush_table()
            in_code = not in_code
            if in_code:
                pdf.subsection("Code Block")
            continue

        if in_code:
            pdf.set_font("Courier", "", 8.5)
            pdf.set_text_color(*pdf.gray_700)
            pdf.multi_cell(0, 5, clean_text(stripped))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_buffer.append(
                [c.strip() for c in clean_text(stripped).split("|")[1:-1]]
            )
            continue
        else:
            flush_table()

        if stripped.startswith("# "):
            pdf.section_band(stripped[2:].strip())
            continue

        if stripped.startswith("## "):
            pdf.section_band(stripped[3:].strip())
            continue

        if stripped.startswith("### "):
            pdf.subsection(stripped[4:].strip())
            continue

        if stripped.startswith("> "):
            pdf.callout_box("Key Insight", stripped[2:].strip(), tone="brand")
            continue

        if stripped in ("---", "***", "___"):
            pdf.divider()
            continue

        # nested bullet
        if line.startswith("  - ") or line.startswith("  * "):
            content = stripped[2:].strip()
            pdf.bullet(content, indent=8)
            continue

        # normal bullet
        if stripped.startswith("- ") or stripped.startswith("* "):
            content = stripped[2:].strip()
            upper = content.upper()

            if "WARNING" in upper:
                pdf.bullet(content.replace("WARNING", "").strip(), color=pdf.warning)
            elif "PASS" in upper:
                pdf.bullet(content.replace("PASS", "").strip(), color=pdf.success)
            elif "FAIL" in upper:
                pdf.bullet(content.replace("FAIL", "").strip(), color=pdf.danger)
            else:
                pdf.bullet(content)
            continue

        pdf.paragraph(stripped)

    flush_table()
    
    # Generate the PDF bytes
    pdf_bytes = bytes(pdf.output())

    # Cleanup temporary image if it was downloaded
    if temp_cover_file and os.path.exists(temp_cover_file):
        try:
            os.remove(temp_cover_file)
        except Exception:
            pass

    return pdf_bytes


def markdown_to_pdf(markdown_text: str, cover_image_path: str | None = None) -> bytes:
    return markdown_to_luxury_pdf(markdown_text, cover_image_path=cover_image_path)

def normalize_markdown_lines(markdown_text: str):
    raw_lines = markdown_text.splitlines()
    normalized = []

    for raw in raw_lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            normalized.append("")
            continue

        # continuation line: attach to previous bullet/paragraph
        is_continuation = (
            raw.startswith("  ")
            or raw.startswith("\t")
        ) and not stripped.startswith(("-", "*", "#", ">", "|", "```"))

        if is_continuation and normalized:
            prev = normalized[-1].rstrip()
            if prev:
                normalized[-1] = prev + " " + stripped
            else:
                normalized.append(stripped)
            continue

        normalized.append(line)

    return normalized    