import os
import re
from bs4 import BeautifulSoup
from typing import List, Tuple
from config import PARSED_TEXT_DIR


def detect_publisher(file_path):
    if file_path.endswith(".html") or file_path.endswith(".htm"):
        return "rsc"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "xml")

    doi_tag = soup.find("article-id", {"pub-id-type": "doi"}) or soup.find("doi")
    if doi_tag:
        doi = doi_tag.get_text(strip=True).lower()
        if doi.startswith("10.1016"):
            return "elsevier"
        elif doi.startswith("10.1021"):
            return "acs"
        elif doi.startswith("10.1007"):
            return "springer"

    content_lower = content.lower()
    if "elsevier" in content_lower or "elsarticle" in content_lower:
        return "elsevier"
    if "acs.org" in content_lower or "american chemical society" in content_lower:
        return "acs"
    if "springer" in content_lower:
        return "springer"

    publisher_tag = soup.find("publisher-name")
    if publisher_tag:
        publisher = publisher_tag.get_text(strip=True).lower()
        if "elsevier" in publisher:
            return "elsevier"
        elif "springer" in publisher:
            return "springer"
        elif "chemical society" in publisher or "acs" in publisher:
            return "acs"

    return "unknown"


def clean_text(text: str) -> str:
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


excluded_keywords = ["supplementary", "reference", "acknowledgement", "acknowledgment", "supporting", "introduction", "interest", "abbreviations", "statement"]


def is_excluded_section(section, publisher):
    if publisher.lower() == "elsevier":
        title_tag = section.find("ce:section-title")
    elif publisher.lower() in ("acs", "springer"):
        title_tag = section.find("title")
    else:
        return False

    if title_tag:
        title_text = title_tag.get_text(strip=True).lower()
        return any(keyword in title_text for keyword in excluded_keywords)
    return False


def process_section(section, seen_texts, body_parts, publisher, only_method=False, parent_title=None):
    if is_excluded_section(section, publisher):
        return

    if publisher.lower() == "elsevier":
        title_tag = section.find("ce:section-title", recursive=False)
        title_text = title_tag.get_text(strip=True) if title_tag else None

        section_key = (title_text or "").lower()
        is_method_section = section_key and ("method" in section_key or "experimental" in section_key)
        if only_method and not is_method_section and not (parent_title and ("method" in parent_title.lower() or "experimental" in parent_title.lower())):
            return

        if title_text:
            body_parts.append(f"[{title_text}]")

        for para in section.find_all(["ce:para", "ce:simple-para"], recursive=False):
            text = para.get_text(strip=True)
            if text and text not in seen_texts:
                seen_texts.add(text)
                body_parts.append(text)

        for child in section.find_all("ce:section", recursive=False):
            process_section(child, seen_texts, body_parts, publisher, only_method, title_text)

    elif publisher.lower() == "acs":
        title_tag = section.find("title", recursive=False)
        title_text = clean_text(title_tag.get_text()) if title_tag else None
        if title_text:
            body_parts.append(f"[{title_text}]")

        exclude_parents = ["caption", "table-wrap-foot", "ack", "fn"]
        exclude_attrs = ["content-type"]

        for para in section.find_all("p", recursive=False):
            parent = para.parent.name if para.parent else ""
            if parent in exclude_parents:
                continue
            if any(attr in para.attrs for attr in exclude_attrs):
                continue
            for tag in para.find_all(["xref", "named-content", "fig", "table-wrap"]):
                tag.decompose()
            text = clean_text(para.get_text())
            if text and text not in seen_texts:
                seen_texts.add(text)
                body_parts.append(text)

        for child in section.find_all("sec", recursive=False):
            process_section(child, seen_texts, body_parts, publisher)

    elif publisher.lower() == "springer":
        title_tag = section.find("title", recursive=False)
        title_text = clean_text(title_tag.get_text()) if title_tag else None
        if title_text:
            body_parts.append(f"[{title_text}]")

        for para in section.find_all("p", recursive=False):
            parent = para.parent.name if para.parent else ""
            if parent in ["notes", "td", "caption", "fig", "th", "table-wrap-foot", "ack", "kwd-group", "ref-list"]:
                continue
            for inner in para.find_all(["fig", "table-wrap"]):
                inner.decompose()
            text = clean_text(para.get_text())
            if text and text not in seen_texts:
                seen_texts.add(text)
                body_parts.append(text)

        for child in section.find_all("sec", recursive=False):
            process_section(child, seen_texts, body_parts, publisher)


def extract_text_from_xml(xml_file_path):
    publisher = detect_publisher(xml_file_path)

    with open(xml_file_path, "r", encoding="utf-8") as file:
        if xml_file_path.endswith(".html") or xml_file_path.endswith(".htm"):
            soup = BeautifulSoup(file, "html.parser")
        else:
            soup = BeautifulSoup(file, "xml")

    title, abstract, body_parts = "", "", []

    if publisher.lower() == "elsevier":
        seen_texts = set()

        title_tag = soup.find("ce:title")
        abstract_tag = soup.find("ce:abstract")
        title = clean_text(title_tag.get_text()) if title_tag else ""
        abstract = clean_text(abstract_tag.get_text()) if abstract_tag else ""

        container = soup.find("ce:sections")
        top_sections = container.find_all("ce:section", recursive=False) if container else soup.find_all("ce:section", recursive=False)

        for section in top_sections:
            process_section(section, seen_texts, body_parts, publisher, only_method=False)

    elif publisher.lower() == "acs":
        seen_texts = set()

        title_tag = soup.find("title-group")
        abstract_tag = soup.find("abstract")
        title = clean_text(title_tag.get_text()) if title_tag else ""
        abstract = clean_text(abstract_tag.get_text()) if abstract_tag else ""

        body_tag = soup.find("body")
        top_sections = body_tag.find_all("sec", recursive=False) if body_tag else soup.find_all("sec", recursive=False)

        for section in top_sections:
            process_section(section, seen_texts, body_parts, publisher)

    elif publisher.lower() == "rsc":
        title_meta = soup.find("meta", {"name": "citation_title"})
        title = clean_text(title_meta["content"]) if title_meta else ""

        abstract_tag = soup.find("p", class_="abstract")
        abstract = clean_text(abstract_tag.get_text(separator=" ")) if abstract_tag else ""

        seen_texts = set()
        exclude_headings = {"acknowledgements", "acknowledgments", "references", "supporting information"}

        for tag in soup.find_all(["span", "p"]):
            cls = tag.get("class") or []

            if "a_heading" in cls or "b_heading" in cls:
                heading_text = clean_text(tag.get_text(separator=" "))
                if heading_text.lower() in exclude_headings:
                    break
                if heading_text:
                    body_parts.append(f"[{heading_text}]")

            elif "otherpara" in cls:
                parts = []
                for child in tag.descendants:
                    if getattr(child, "name", None) is None:
                        parent_cls = child.parent.get("class") or [] if child.parent else []
                        if not any(c in parent_cls for c in ["sup_ref", "ref"]):
                            parts.append(str(child))
                text = clean_text(" ".join(parts))
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    body_parts.append(text)

    elif publisher.lower() == "springer":
        title_tag = soup.find("article-title")
        title = clean_text(title_tag.get_text()) if title_tag else ""

        abstract_tag = soup.find("abstract")
        abstract = ""
        if abstract_tag:
            abstract_parts = []
            abstract_title = abstract_tag.find("title")
            if abstract_title:
                abstract_parts.append(clean_text(abstract_title.get_text()))
            for para in abstract_tag.find_all("p"):
                abstract_parts.append(clean_text(para.get_text()))
            abstract = " ".join(abstract_parts)

        seen_texts = set()

        body_tag = soup.find("body")
        top_sections = body_tag.find_all("sec", recursive=False) if body_tag else soup.find_all("sec", recursive=False)

        for section in top_sections:
            process_section(section, seen_texts, body_parts, publisher)

    body = "\n".join([clean_text(t) for t in body_parts if t])
    return f"{title}\n\n{abstract}\n\n{body}"


def parse_all_xml_files(xml_dir: str) -> int:
    os.makedirs(PARSED_TEXT_DIR, exist_ok=True)
    parsed_counts = 0

    for filename in os.listdir(xml_dir):
        if not filename.endswith(".xml") and not filename.endswith(".html"):
            continue

        file_path = os.path.join(xml_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(PARSED_TEXT_DIR, output_filename)

        if os.path.exists(output_path):
            print(f"Skipped (already exists): {output_filename}")
            continue

        parsed_text = extract_text_from_xml(file_path)
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(parsed_text)

        parsed_counts += 1
        print(f"Parsed and saved: {output_filename}")

    print(f"Parsed {parsed_counts} new documents.")
    return parsed_counts


def delete_short_txt_files(directory, min_lines=10):
    deleted_files = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                if len(f.readlines()) < min_lines:
                    os.remove(file_path)
                    deleted_files += 1
                    print(f"Deleted (too short): {filename}")
    print(f"Deleted {deleted_files} file(s) with fewer than {min_lines} lines.")


SECTION_RE = re.compile(r'^\s*\[([^\]]+)\]\s*$', re.MULTILINE)
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _clean_noise(text: str) -> str:
    text = re.sub(r'\\documentclass.*?\\end\{document\}', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _split_by_sections(text: str) -> List[Tuple[str, str]]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("BODY", text)]

    parts: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((title, text[start:end].strip()))
    return parts


def chunk_text(
    text: str,
    max_chars: int = 1500,
    min_chars: int = 400,
    overlap_sents: int = 1,
    exclude_sections: Tuple[str, ...] = ("references",),
) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    chunks: List[str] = []
    carry = ""

    for sec_title, sec_body in _split_by_sections(text):
        if any(x in sec_title.lower().strip() for x in exclude_sections):
            continue

        body = _clean_noise(sec_body)
        if not body:
            continue

        sents = [s.strip() for s in SENT_SPLIT_RE.split(body) if s.strip()]
        cur_sents: List[str] = []
        cur_len = 0

        def flush_current():
            nonlocal carry, cur_sents, cur_len, chunks
            if not cur_sents:
                return

            text_block = f"[{sec_title}] " + " ".join(cur_sents)
            text_block = text_block.strip()

            if len(text_block) < min_chars:
                carry = (carry + " " + text_block).strip() if carry else text_block
            else:
                if carry:
                    text_block = (carry + " " + text_block).strip()
                    carry = ""
                chunks.append(text_block)

            if overlap_sents > 0 and cur_sents:
                cur_sents = cur_sents[-overlap_sents:]
                cur_len = sum(len(s) + 1 for s in cur_sents)
            else:
                cur_sents = []
                cur_len = 0

        for s in sents:
            add_len = len(s) + 1
            if cur_len + add_len <= max_chars:
                cur_sents.append(s)
                cur_len += add_len
            else:
                flush_current()
                if cur_len + add_len > max_chars and not cur_sents:
                    chunks.append(f"[{sec_title}] " + s[:max_chars].strip())
                    cur_sents, cur_len = [], 0
                else:
                    cur_sents.append(s)
                    cur_len += add_len

        flush_current()

    if carry:
        chunks.append(carry)

    return chunks


if __name__ == "__main__":
    from config import RAW_XML_DIR
    parse_all_xml_files(RAW_XML_DIR)
