"""Versioned deterministic HTML and boilerplate cleaner for product descriptions (desc_clean_v1)."""

from dataclasses import dataclass, field
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unicodedata

import pandas as pd

from recommender.artifacts import json_digest
from recommender.experiments.static_enrichment.config import DEFAULT_DESCRIPTION_PATH
from recommender.experiments.static_enrichment.metadata import clean_html_description
from recommender.merging import merge_records

CLEANER_VERSION = "desc_clean_v1"

CLEANING_RULES: Dict[str, Any] = {
    "version": CLEANER_VERSION,
    "csv_columns_used": [0, 1, 2],
    "csv_columns_excluded": [3, 4],
    "rules": [
        {
            "id": "R1_html_structure_and_hidden_removal",
            "description": (
                "Parse HTML with stdlib HTMLParser; strip tags, attributes, <script>, <style>, <noscript>, "
                "<template>, HTML comments, <img> elements, and hidden elements (hidden, aria-hidden=true, "
                "display:none, visibility:hidden). Preserve paragraph/heading/list block breaks and convert "
                "tables into Header: Value | Header2: Value2 (or Cell | Cell2) pairs."
            ),
        },
        {
            "id": "R2_entities_encoded_markup_unicode_nfc",
            "description": (
                "Selectively decode genuine encoded HTML markup before parsing without mistaking numeric "
                "comparisons (<1%, >200°C) for HTML tags; decode HTML entities; normalize Unicode to NFC; "
                "convert NBSP/figure/narrow spaces to ASCII spaces; remove zero-width/format chars (Cf) and "
                "C0/C1 control characters."
            ),
        },
        {
            "id": "R3_decorative_emoji_and_bullet_cleanup",
            "description": (
                "Convert keycap step emojis (1️⃣) to numbered markers (1.); remove decorative emojis, "
                "checkmarks (✅, 🔹, 📌, 🔧, 🚗, etc.), variation selectors, and Quill leading bullet dots "
                "while preserving technical symbols (°, ±, ≈, ≤, ≥, ×, ®, ™, №, %, /, -, :)."
            ),
        },
        {
            "id": "R4_conservative_boilerplate_removal",
            "description": (
                "Remove non-technical CTA headings (e.g. 'Чому варто купити ...?', 'Як перевірити оригінальність...?'), "
                "authenticity/QR/URL/email/phone/shipping/payment/cart notices, and pure price/promotional fluff "
                "sentences while preserving all technical paragraphs, viscosity/ISO/NLGI grades, API/ACEA/OEM "
                "approvals, formally-approved vs meets-requirements distinctions, temperatures, ratios, and negations "
                "such as 'не містить'."
            ),
        },
        {
            "id": "R5_within_product_exact_dedup",
            "description": (
                "Deduplicate exact repeated blocks and sentences (case-insensitive, whitespace-normalized) within "
                "each individual product while preserving first occurrence and performing zero cross-product dedup."
            ),
        },
    ],
}
CLEANING_RULES_HASH = json_digest(CLEANING_RULES)

_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr",
})
_SKIP_CONTENT_TAGS = frozenset({"script", "style", "noscript", "template"})
_BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol",
})

_ENCODED_HTML_TAG_RE = re.compile(
    r"&lt;/?(?:p|br|div|span|strong|b|i|em|u|ul|ol|li|table|tbody|thead|tr|td|th|h[1-6]|style|script|a|img|section|article|blockquote)\b[^&]{0,300}&gt;|&lt;!--",
    re.IGNORECASE,
)
_DOUBLE_ENCODED_TAG_RE = re.compile(r"&amp;(?:lt|gt|nbsp|amp);", re.IGNORECASE)
_HIDDEN_STYLE_RE = re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)

_KEYCAP_EMOJI_RE = re.compile(r"([0-9])\ufe0f?\u20e3")
_EMOJI_AND_DECOR_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "\u2300-\u23FF"
    "\u2B00-\u2BFF"
    "\U0001F000-\U0001F02F"
    "\ufe0e\ufe0f"
    "]+",
    flags=re.UNICODE,
)
_LEADING_BULLET_RE = re.compile(r"^\s*[·•▪▫◦‣⁃]+\s*")
_MULTISPACE_RE = re.compile(r"[ \t\r\f\v]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

# Sentence boundary: split after . ! ? when followed by whitespace and uppercase/digit/quote/Ukrainian letter,
# avoiding splits inside decimal numbers (5.40, 229.3) or abbreviations like "об.%".
_SENTENCE_SPLIT_RE = re.compile(r"(?<!\d)(?<=[.!?])\s+(?=[\"'«(A-ZА-ЯІЇЄҐ0-9])")

# Conservative boilerplate patterns
_CTA_HEADING_RE = re.compile(
    r"^(?:"
    r"чому\s+(?:варто\s+)?(?:купити|обрати|обирають|всі\s+купують)\b"
    r"|як\s+перевірити\s+оригінальність\b"
    r"|де\s+(?:купити|замовити)\b"
    r"|доставка\s+(?:та|і)\s+оплата\b"
    r"|оформлення\s+замовлення\b"
    r"|контакт(?:и|на\s+інформація)\b"
    r")",
    re.IGNORECASE,
)

_URL_EMAIL_PHONE_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+|\+?380[\s()-]*\d{2}[\s()-]*\d{3}[\s()-]*\d{2}[\s()-]*\d{2})",
    re.IGNORECASE,
)

_EXPLICIT_LOGISTICS_AUTH_RE = re.compile(
    r"(?:"
    r"(?:перевірити|переконатися\s+в)\s+оригінальност[іи].{0,100}(?:shellcheck|qr-?код)"
    r"|доставка\s+(?:по\s+(?:всій\s+)?україні|новою\s+поштою|укрпоштою|кур['’]?єром)"
    r"|оплата\s+(?:при\s+отриманні|готівкою|карткою|безготівков|на\s+рахунок)"
    r"|(?:додати|додайте)\s+(?:у|в)\s+кошик"
    r"|оформит[иь]\s+замовлення"
    r"|замов(?:ляйте|ити)\s+(?:онлайн|зараз|на\s+сайті|за\s+телефоном)"
    r"|(?:зв['’]?яжіться\s+з\s+менеджером|безкоштовна\s+консультація|телефонуйте)"
    r"|наш(?:ому|ого)\s+інтернет-?магазин"
    r")",
    re.IGNORECASE,
)

_PURE_MARKETING_SENT_RE = re.compile(
    r"(?:"
    r"^(?:доступн[аией]+\s+в\s+україні\s*[–—-]|один\s+із\s+найкращих\s+варіантів\s+за\s+співвідношенням\s+ціна/якість)"
    r"|^ціна\s+синтетичної\s+моторної\s+оливи\s+залежить\s+від\s+літражу"
    r"|^якщо\s+ви\s+шукаєте\s+кращу\s+моторну\s+оливу\s+за\s+розумною\s+ціною"
    r"|^популярність\s+цієї\s+оливи\s+зумовлена\s+поєднанням"
    r"|^перевірена\s+роками\s+та\s+популярна\s+серед\s+сто"
    r"|^чому\s+.{1,80}\s+кращ[аеі]\s+за\s+інші\s+оливи\??$"
    r"|(?:^(?:це\s+)?|(?:^|\s)[–—-]\s*(?:це\s+)?)(?:вибір\s+професіоналів|вибір\s+водіїв|найкращий\s+вибір|найкращий\s+помічник|вибір\s*№\s*1|ваш\s+вибір|надійний\s+вибір|ідеальний\s+вибір|ідеальне\s+рішення|ідеальний\s+засіб|надійний\s+помічник|перевірена\s+якіст[ьі])\b[^.]*!$"
    r")",
    re.IGNORECASE,
)

_TECHNICAL_SIGNAL_RE = re.compile(
    r"(?:"
    r"\b\d+W-?\d+\b"
    r"|\b(?:API|ACEA|ILSAC|JASO|ISO|NLGI|DIN|ASTM|SAE|DOT|BS|NATO|MB|VW|BMW|GM|PSA|MAN|FORD|RENAULT|FIAT|FCA|CUMMINS|DEUTZ|VOLVO|MACK|DAF|OPEL|LSPI|DPF|NOACK|OAT|NAP-free|AdBlue)\b"
    r"|-?\d+(?:[.,]\d+)?\s*°\s*[CF]?"
    r"|\b\d+\s*:\s*\d+\b"
    r"|\bне\s+містить\b"
    r"|\bбез\s+(?:сажов|силікон|фосфат|нітрит|амін|метанол)"
    r")",
    re.IGNORECASE,
)

_SPEC_AUDIT_RE = re.compile(
    r"(?:"
    r"\b\d+W-?\d+\b"
    r"|\b(?:API|ACEA|ILSAC|JASO)\s+[A-Z0-9/+.-]+"
    r"|\b(?:ISO|NLGI|DIN|ASTM|SAE|DOT)\s+[A-Z0-9/+.-]+"
    r"|\b(?:MB|VW|BMW|RN|WSS|TL|GMW|DEXOS)\s*[A-Z0-9./-]+"
    r"|-?\d+(?:[.,]\d+)?\s*°\s*[CF]?"
    r"|\b\d+\s*:\s*\d+\b"
    r"|\bне\s+містить\b"
    r")",
    re.IGNORECASE,
)


def extract_spec_tokens(text: str) -> set[str]:
    """Extract normalized key technical specification tokens for retention verification."""
    if not text:
        return set()
    tokens = set()
    for m in _SPEC_AUDIT_RE.finditer(text):
        tok = re.sub(r"\s+", " ", m.group(0)).strip().rstrip(".,;:)]").casefold()
        if tok:
            tokens.add(tok)
    return tokens


def _normalize_unicode_and_controls(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    out_chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if code in (0x00A0, 0x2007, 0x202F):
            out_chars.append(" ")
            continue
        if ch in ("\n", "\r", "\t"):
            out_chars.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat == "Cf":
            # Strip zero-width spaces, joiners, soft hyphens, BOM, directional marks
            continue
        if cat == "Cc" or (0x00 <= code <= 0x1F) or (0x7F <= code <= 0x9F):
            # Strip C0/C1 control characters
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def _decode_encoded_markup_if_present(text: str) -> tuple[str, int]:
    """Decode HTML-encoded tag markup safely without converting math comparisons like &lt;1% into tags."""
    passes = 0
    for _ in range(2):
        if _DOUBLE_ENCODED_TAG_RE.search(text) or _ENCODED_HTML_TAG_RE.search(text):
            unescaped = html.unescape(text)
            if unescaped == text:
                break
            text = unescaped
            passes += 1
        else:
            break
    return text, passes


class _StructuredHTMLParser(HTMLParser):
    """Extract readable blocks and structured table pairs while dropping hidden/script/style/img elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []  # (kind, text)
        self._buf: list[str] = []
        self._tag_stack: list[tuple[str, bool, bool]] = []  # (tag, is_skip, is_hidden)
        self._skip_depth = 0
        self._hidden_depth = 0
        self._in_heading = False
        self._in_list_item = False

        # Table parsing state
        self._in_table = False
        self._table_rows: list[list[tuple[str, bool]]] = []
        self._current_row: list[tuple[str, bool]] = []
        self._in_cell = False
        self._cell_buf: list[str] = []
        self._cell_is_header_tag = False
        self._cell_has_strong = False
        self._cell_has_plain_text = False

        # Stats
        self.tags_stripped = 0
        self.hidden_or_script_blocks_dropped = 0
        self.tables_parsed = 0

    def _flush_text_block(self, kind: str = "p") -> None:
        raw = "".join(self._buf)
        self._buf.clear()
        txt = _MULTISPACE_RE.sub(" ", raw.replace("\n", " ")).strip()
        if txt:
            self.blocks.append((kind, txt))

    def _is_hidden_attrs(self, attrs: list[tuple[str, Optional[str]]]) -> bool:
        for k, v in attrs:
            kl = k.lower()
            vl = (v or "").strip().lower()
            if kl == "hidden":
                return True
            if kl == "aria-hidden" and vl in ("true", "1", "yes"):
                return True
            if kl == "style" and _HIDDEN_STYLE_RE.search(vl):
                return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tl = tag.lower()
        self.tags_stripped += 1
        is_skip = tl in _SKIP_CONTENT_TAGS
        is_hidden = self._is_hidden_attrs(attrs)

        if tl not in _VOID_TAGS:
            self._tag_stack.append((tl, is_skip, is_hidden))
            if is_skip:
                self._skip_depth += 1
                self.hidden_or_script_blocks_dropped += 1
            if is_hidden:
                self._hidden_depth += 1
                self.hidden_or_script_blocks_dropped += 1

        if self._skip_depth > 0 or self._hidden_depth > 0:
            return

        if tl == "br":
            if self._in_cell:
                self._cell_buf.append(" ")
            else:
                self._flush_text_block("h" if self._in_heading else ("li" if self._in_list_item else "p"))
            return
        if tl == "img":
            return

        if tl == "table":
            self._flush_text_block()
            self._in_table = True
            self._table_rows = []
            return
        if self._in_table:
            if tl == "tr":
                self._current_row = []
            elif tl in ("td", "th"):
                self._in_cell = True
                self._cell_buf = []
                self._cell_is_header_tag = tl == "th"
                self._cell_has_strong = False
                self._cell_has_plain_text = False
            elif tl in ("strong", "b") and self._in_cell:
                self._cell_has_strong = True
            return

        if tl in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush_text_block()
            self._in_heading = True
        elif tl == "li":
            self._flush_text_block()
            self._in_list_item = True
        elif tl in _BLOCK_TAGS:
            self._flush_text_block("h" if self._in_heading else ("li" if self._in_list_item else "p"))

    def handle_endtag(self, tag: str) -> None:
        tl = tag.lower()
        if tl in _VOID_TAGS:
            return

        was_skip_or_hidden = self._skip_depth > 0 or self._hidden_depth > 0
        # Pop matching tag from stack safely even on malformed HTML
        for idx in range(len(self._tag_stack) - 1, -1, -1):
            st_tag, st_skip, st_hidden = self._tag_stack[idx]
            if st_tag == tl:
                for _, p_skip, p_hidden in self._tag_stack[idx:]:
                    if p_skip and self._skip_depth > 0:
                        self._skip_depth -= 1
                    if p_hidden and self._hidden_depth > 0:
                        self._hidden_depth -= 1
                del self._tag_stack[idx:]
                break

        if was_skip_or_hidden:
            return

        if self._in_table:
            if tl in ("td", "th") and self._in_cell:
                cell_txt = _MULTISPACE_RE.sub(" ", "".join(self._cell_buf).replace("\n", " ")).strip()
                is_hdr = self._cell_is_header_tag or (self._cell_has_strong and not self._cell_has_plain_text)
                self._current_row.append((cell_txt, is_hdr))
                self._in_cell = False
                self._cell_buf = []
            elif tl == "tr":
                if any(txt for txt, _ in self._current_row):
                    self._table_rows.append(self._current_row)
                self._current_row = []
            elif tl == "table":
                self._emit_table()
                self._in_table = False
            return

        if tl in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush_text_block("h")
            self._in_heading = False
        elif tl == "li":
            self._flush_text_block("li")
            self._in_list_item = False
        elif tl in _BLOCK_TAGS:
            self._flush_text_block("p")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0 or self._hidden_depth > 0:
            return
        if self._in_cell:
            self._cell_buf.append(data)
            if data.strip() and not any(t in ("strong", "b") for t, _, _ in self._tag_stack):
                self._cell_has_plain_text = True
        else:
            self._buf.append(data)

    def handle_comment(self, data: str) -> None:
        self.hidden_or_script_blocks_dropped += 1

    def _emit_table(self) -> None:
        if not self._table_rows:
            return
        self.tables_parsed += 1
        rows = self._table_rows
        self._table_rows = []
        first_row = rows[0]
        use_header = (
            len(rows) >= 2
            and len(first_row) >= 2
            and all(is_hdr for txt, is_hdr in first_row if txt)
            and all(len(r) == len(first_row) for r in rows[1:])
        )
        if use_header:
            headers = [txt for txt, _ in first_row]
            for r in rows[1:]:
                pairs = []
                for hdr, (val, _) in zip(headers, r):
                    if hdr and val:
                        pairs.append(f"{hdr}: {val}")
                    elif val:
                        pairs.append(val)
                if pairs:
                    self.blocks.append(("table_row", " | ".join(pairs)))
        else:
            for r in rows:
                cells = [txt for txt, _ in r if txt]
                if cells:
                    self.blocks.append(("table_row", " | ".join(cells)))

    def finalize(self) -> None:
        if self._in_table:
            if self._in_cell:
                cell_txt = _MULTISPACE_RE.sub(" ", "".join(self._cell_buf).replace("\n", " ")).strip()
                self._current_row.append((cell_txt, self._cell_is_header_tag))
                self._in_cell = False
            if any(txt for txt, _ in self._current_row):
                self._table_rows.append(self._current_row)
            self._emit_table()
            self._in_table = False
        self._flush_text_block("h" if self._in_heading else ("li" if self._in_list_item else "p"))


def _clean_surface_text(text: str) -> tuple[str, int]:
    """Normalize whitespace/bullets and remove decorative emojis while preserving technical symbols."""
    text = _normalize_unicode_and_controls(text)
    text = _KEYCAP_EMOJI_RE.sub(r"\1. ", text)
    emoji_matches = _EMOJI_AND_DECOR_RE.findall(text)
    emojis_removed = len(emoji_matches)
    if emoji_matches:
        text = _EMOJI_AND_DECOR_RE.sub(" ", text)
    text = _LEADING_BULLET_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text, emojis_removed


def _strip_inline_url_or_phone_only_if_safe(sentence: str) -> Optional[str]:
    """Return None to drop pure logistics/URL/phone/marketing sentences; otherwise keep sentence."""
    if _URL_EMAIL_PHONE_RE.search(sentence):
        return None
    if _EXPLICIT_LOGISTICS_AUTH_RE.search(sentence):
        return None
    if _PURE_MARKETING_SENT_RE.search(sentence):
        # Conservative guard: keep if sentence has real technical approvals/measurements/negations
        # beyond a simple product viscosity title.
        stripped_name = re.sub(r"\b\d+W-?\d+\b", "", sentence)
        if not _TECHNICAL_SIGNAL_RE.search(stripped_name):
            return None
    return sentence


def clean_description_v1(
    raw: object,
    *,
    return_stats: bool = False,
) -> str | tuple[str, Dict[str, Any]]:
    """Deterministically clean HTML product description into structured, spec-preserving text."""
    stats: Dict[str, Any] = {
        "cleaner_version": CLEANER_VERSION,
        "raw_chars": 0,
        "cleaned_chars": 0,
        "encoded_markup_passes": 0,
        "tags_stripped": 0,
        "hidden_or_script_blocks_dropped": 0,
        "tables_parsed": 0,
        "emojis_removed": 0,
        "boilerplate_blocks_removed": 0,
        "boilerplate_sentences_removed": 0,
        "duplicate_blocks_removed": 0,
        "duplicate_sentences_removed": 0,
        "removed_items": [],
    }
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or pd.isna(raw):
        return ("", stats) if return_stats else ""

    text = str(raw).strip()
    if not text or text.upper() == "NULL":
        return ("", stats) if return_stats else ""

    stats["raw_chars"] = len(text)
    text = _normalize_unicode_and_controls(text)
    text, enc_passes = _decode_encoded_markup_if_present(text)
    stats["encoded_markup_passes"] = enc_passes

    parser = _StructuredHTMLParser()
    parser.feed(text)
    parser.close()
    parser.finalize()

    stats["tags_stripped"] = parser.tags_stripped
    stats["hidden_or_script_blocks_dropped"] = parser.hidden_or_script_blocks_dropped
    stats["tables_parsed"] = parser.tables_parsed

    seen_blocks: set[str] = set()
    seen_sentences: set[str] = set()
    kept_blocks: list[str] = []

    for kind, raw_block in parser.blocks:
        cleaned_block, n_emojis = _clean_surface_text(raw_block)
        stats["emojis_removed"] += n_emojis
        if not cleaned_block:
            continue

        # Block-level CTA heading filter
        if _CTA_HEADING_RE.search(cleaned_block):
            stats["boilerplate_blocks_removed"] += 1
            stats["removed_items"].append({"reason": "cta_heading", "text": cleaned_block})
            continue

        # Exact duplicate block check (within product)
        block_key = cleaned_block.casefold()
        if block_key in seen_blocks:
            stats["duplicate_blocks_removed"] += 1
            stats["removed_items"].append({"reason": "duplicate_block", "text": cleaned_block})
            continue

        if kind == "table_row":
            seen_blocks.add(block_key)
            kept_blocks.append(cleaned_block)
            continue

        # Sentence-level conservative boilerplate and exact duplicate sentence dedup within product
        sentences = _SENTENCE_SPLIT_RE.split(cleaned_block)
        kept_sents: list[str] = []
        for sent in sentences:
            s_clean = sent.strip()
            if not s_clean:
                continue
            kept_s = _strip_inline_url_or_phone_only_if_safe(s_clean)
            if kept_s is None:
                stats["boilerplate_sentences_removed"] += 1
                stats["removed_items"].append({"reason": "boilerplate_sentence", "text": s_clean})
                continue
            sent_key = kept_s.casefold()
            if sent_key in seen_sentences:
                stats["duplicate_sentences_removed"] += 1
                stats["removed_items"].append({"reason": "duplicate_sentence", "text": kept_s})
                continue
            seen_sentences.add(sent_key)
            kept_sents.append(kept_s)

        if not kept_sents:
            continue

        final_block = " ".join(kept_sents)
        final_key = final_block.casefold()
        if final_key in seen_blocks:
            stats["duplicate_blocks_removed"] += 1
            stats["removed_items"].append({"reason": "duplicate_block", "text": final_block})
            continue
        seen_blocks.add(final_key)
        kept_blocks.append(final_block)

    result = "\n".join(kept_blocks).strip()
    stats["cleaned_chars"] = len(result)
    return (result, stats) if return_stats else result


def load_descriptions_v1(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Read ONLY columns 0/1/2 (name, product_id, raw_description); never read/append price columns 3/4."""
    source = Path(path) if path is not None else DEFAULT_DESCRIPTION_PATH
    raw = pd.read_csv(
        source,
        sep=";",
        header=None,
        usecols=[0, 1, 2],
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    if not {0, 1, 2}.issubset(raw.columns):
        raise ValueError("Description export must have columns 0, 1, 2 (name, product_id, html_description).")
    out = pd.DataFrame({
        "raw_name": raw[0].astype(str).str.strip(),
        "product_id": raw[1].astype(str).str.strip(),
        "raw_description": raw[2].astype(str),
    })
    out = out.loc[~out["product_id"].isin(["", "NULL"])].copy()
    out["legacy_html_description"] = out["raw_description"].map(clean_html_description)
    cleaned_vals: list[str] = []
    cleaning_stats: list[Dict[str, Any]] = []
    for raw_desc in out["raw_description"].tolist():
        c_txt, c_stat = clean_description_v1(raw_desc, return_stats=True)
        cleaned_vals.append(c_txt)
        cleaning_stats.append(c_stat)
    out["cleaned_description_v1"] = cleaned_vals
    out["cleaning_stats_json"] = [json.dumps(s, ensure_ascii=False) for s in cleaning_stats]
    out = merge_records(
        out[["product_id", "raw_name", "raw_description", "legacy_html_description", "cleaned_description_v1", "cleaning_stats_json"]],
        keys=["product_id"],
    )
    return out
