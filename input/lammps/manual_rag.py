from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "input" / "manual_rag_corpus" / "lammps"
COMMAND_SUMMARY_PATH = CORPUS_DIR / "lammps_command_summaries.v1.jsonl"
PURPOSE_CHUNK_PATH = CORPUS_DIR / "lammps_purpose_chunks.v1.jsonl"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./+-]*")
COMMAND_ALIASES = {
    "fix npt": {"fix nvt"},
    "fix nph": {"fix nvt"},
}
COMMON_BOILERPLATE_COMMANDS = {
    "dump",
    "dump_modify",
    "fix npt",
    "fix nve",
    "fix nvt",
    "group",
    "minimize",
    "min_style",
    "neighbor",
    "neigh_modify",
    "run",
    "thermo",
    "thermo_style",
    "timestep",
    "undump",
    "unfix",
    "velocity",
}
GENERIC_CO_OCCURRING_COMMANDS = COMMON_BOILERPLATE_COMMANDS | {
    "atom_style",
    "compute",
    "create_atoms",
    "create_box",
    "fix",
    "if",
    "label",
    "lattice",
    "log",
    "mass",
    "pair_coeff",
    "pair_style",
    "print",
    "region",
    "set",
    "units",
    "variable",
}
SUPPORTING_COMMON_PRIORITY = [
    "fix nvt",
    "fix npt",
    "fix nve",
    "velocity",
    "thermo_style",
    "timestep",
    "run",
    "dump",
    "dump_modify",
    "neighbor",
    "neigh_modify",
    "minimize",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
    "generate",
    "job",
    "lammps",
    "mof",
    "only",
    "section",
    "using",
    "write",
    "writes",
}
PURPOSE_ALLOWED_FAMILIES = {"command", "compute", "dump", "fix", "howto", "kspace", "official_example"}
PURPOSE_NOISY_PAGES = {"Howto_output.html", "Howto_structured_data.html"}
PURPOSE_SELECTOR_MAX_CANDIDATES = int(os.getenv("SIMMOF_LAMMPS_RAG_SELECTOR_MAX_CANDIDATES", "160"))
PURPOSE_SELECTOR_MAX_PROMPT_CHARS = int(os.getenv("SIMMOF_LAMMPS_RAG_SELECTOR_MAX_PROMPT_CHARS", "48000"))
PURPOSE_SELECTOR_LEXICAL_TOP_K = int(os.getenv("SIMMOF_LAMMPS_RAG_SELECTOR_LEXICAL_TOP_K", "80"))
EVIDENCE_MAX_CORE_CHUNKS = int(os.getenv("SIMMOF_LAMMPS_RAG_MAX_CORE_CHUNKS", "5"))
EVIDENCE_MAX_SUPPORT_CHUNKS = int(os.getenv("SIMMOF_LAMMPS_RAG_MAX_SUPPORT_CHUNKS", "2"))
EVIDENCE_MAX_DEPENDENCY_COMMANDS = int(
    os.getenv("SIMMOF_LAMMPS_RAG_MAX_DEPENDENCY_COMMANDS", "4")
)
COMMAND_EVIDENCE_MAX_HITS = int(os.getenv("SIMMOF_LAMMPS_COMMAND_EVIDENCE_MAX_HITS", "7"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokens(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _score(query_tokens: List[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for token in text_tokens:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    for token in query_tokens:
        tf = counts.get(token, 0)
        if tf:
            score += 1.0 + math.log(tf)
    return score / math.sqrt(len(text_tokens))


def _bm25_rank(
    rows: Iterable[Dict[str, Any]],
    query: str,
    fields: Tuple[str, ...],
    top_k: int,
    *,
    family_filter: Iterable[str] = (),
) -> List[Tuple[float, Dict[str, Any]]]:
    candidates: List[Tuple[Dict[str, Any], List[str], str]] = []
    allowed = {str(x) for x in family_filter if x}
    for row in rows:
        if allowed and str(row.get("family")) not in allowed:
            continue
        haystack = " ".join(str(row.get(field, "")) for field in fields)
        tokens = _tokens(haystack)
        if not tokens:
            continue
        candidates.append((row, tokens, haystack))

    query_tokens = list(dict.fromkeys(_tokens(query)))
    if not candidates or not query_tokens:
        return []

    df: Dict[str, int] = {}
    for _row, tokens, _haystack in candidates:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    avgdl = sum(len(tokens) for _row, tokens, _haystack in candidates) / len(candidates)
    n_docs = len(candidates)
    k1 = 1.5
    b = 0.75

    scored: List[Tuple[float, Dict[str, Any]]] = []
    query_lower = query.lower()
    for row, tokens, haystack in candidates:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        dl = len(tokens)
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if not tf:
                continue
            idf = math.log(1 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

        title = str(row.get("title") or row.get("purpose_text") or row.get("anchor_command") or "").lower()
        anchor = str(row.get("anchor_command") or "").lower()
        if anchor and anchor in query_lower:
            score += 3.0
        for phrase in re.findall(r"[a-z0-9][a-z0-9 /+-]{3,}", title):
            phrase = phrase.strip()
            if len(phrase) > 5 and phrase in query_lower:
                score += 2.0
        if str(row.get("page_name")) in PURPOSE_NOISY_PAGES:
            score *= 0.55
        if score > 0:
            out = dict(row)
            out["score"] = round(score, 6)
            scored.append((score, out))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def _compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _parse_selector_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"selected_chunk_ids": [], "support_chunk_ids": [], "rationale": text[:1000]}


def load_lammps_command_summaries() -> List[Dict[str, Any]]:
    return _load_jsonl(COMMAND_SUMMARY_PATH)


def load_lammps_purpose_chunks() -> List[Dict[str, Any]]:
    return _load_jsonl(PURPOSE_CHUNK_PATH)


def _rows_by_command_name(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        command_name = str(row.get("command_name", "")).lower().strip()
        if command_name and command_name not in out:
            out[command_name] = row
    return out


def _supporting_common_hits(
    rows: Iterable[Dict[str, Any]],
    required_commands: Iterable[str],
    max_hits: int,
) -> List[Dict[str, Any]]:
    if max_hits <= 0:
        return []
    row_map = _rows_by_command_name(rows)
    required = {str(command).lower().strip() for command in required_commands if command}
    selected: List[Dict[str, Any]] = []
    for command in SUPPORTING_COMMON_PRIORITY:
        if command not in required:
            continue
        row = row_map.get(command)
        if not row:
            for alias in COMMAND_ALIASES.get(command, set()):
                row = row_map.get(alias)
                if row:
                    break
        if not row:
            continue
        out = dict(row)
        out["score"] = "support"
        selected.append(out)
        if len(selected) >= max_hits:
            break
    return selected


def _rank(
    rows: Iterable[Dict[str, Any]],
    query: str,
    fields: Tuple[str, ...],
    top_k: int,
    *,
    required_commands: Iterable[str] = (),
    exclude_common_boilerplate: bool = True,
) -> List[Dict[str, Any]]:
    query_tokens = _tokens(query)
    required = {str(command).lower().strip() for command in required_commands if command}
    property_specific_required = required - COMMON_BOILERPLATE_COMMANDS
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        haystack = " ".join(str(row.get(field, "")) for field in fields)
        score = _score(query_tokens, haystack)
        command_name = str(row.get("command_name", "")).lower().strip()
        if command_name.endswith(" styles"):
            continue
        if exclude_common_boilerplate and command_name in COMMON_BOILERPLATE_COMMANDS:
            continue
        if command_name in property_specific_required:
            score += 7.0
        if command_name == "fix nvt" and ({"fix npt", "fix nph"} & property_specific_required):
            score += 7.0
        haystack_lower = haystack.lower()
        for command in property_specific_required:
            if command and command in haystack_lower and command != command_name:
                score += 2.5
                break
        if score > 0:
            out = dict(row)
            out["score"] = round(score, 6)
            scored.append((score, out))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def retrieve_lammps_command_hints(
    query: str,
    *,
    required_commands: Iterable[str] = (),
    top_k: int = 12,
    max_chars_per_hit: int = 900,
    exclude_common_boilerplate: bool = True,
    max_common_support_hits: int = 2,
) -> Dict[str, Any]:
    rows = load_lammps_command_summaries()
    hits = _rank(
        rows,
        query,
        fields=(
            "command_name",
            "family",
            "syntax",
            "short_summary",
            "output_info_summary",
            "restrictions_summary",
            "default_summary",
            "examples",
            "related_commands",
        ),
        top_k=top_k,
        required_commands=required_commands,
        exclude_common_boilerplate=exclude_common_boilerplate,
    )
    required_order = [
        str(command).lower().strip()
        for command in required_commands
        if command
        and (
            not exclude_common_boilerplate
            or str(command).lower().strip() not in COMMON_BOILERPLATE_COMMANDS
        )
    ]
    required_normalized = set(required_order)

    present = {str(hit.get("command_name", "")).lower().strip() for hit in hits}
    for command in required_order:
        if command in present:
            continue
        acceptable_names = {command} | COMMAND_ALIASES.get(command, set())
        if acceptable_names & present:
            continue
        forced = None
        for row in rows:
            command_name = str(row.get("command_name", "")).lower().strip()
            if command_name in acceptable_names:
                forced = dict(row)
                forced["score"] = "forced"
                break
        if forced:
            hits.append(forced)
            present.add(str(forced.get("command_name", "")).lower().strip())

    support_hits = (
        _supporting_common_hits(rows, required_commands, max_common_support_hits)
        if exclude_common_boilerplate
        else []
    )

    def format_hit(hit: Dict[str, Any]) -> str:
        syntax = "; ".join(hit.get("syntax") or [])
        examples = "; ".join(hit.get("examples") or [])
        related = ", ".join(hit.get("related_commands") or [])
        body = " ".join(
            part
            for part in [
                f"syntax: {syntax}" if syntax else "",
                f"summary: {hit.get('short_summary', '')}" if hit.get("short_summary") else "",
                f"output: {hit.get('output_info_summary', '')}" if hit.get("output_info_summary") else "",
                f"restrictions: {hit.get('restrictions_summary', '')}" if hit.get("restrictions_summary") else "",
                f"default: {hit.get('default_summary', '')}" if hit.get("default_summary") else "",
                f"examples: {examples}" if examples else "",
                f"related: {related}" if related else "",
            ]
            if part
        )
        return (
            f"- {hit.get('command_name')} [{hit.get('family')}; score={hit.get('score')}; "
            f"{hit.get('source_url')}]: {_compact(body, max_chars_per_hit)}"
        )

    lines: List[str] = []
    if hits or support_hits:
        lines.append("[LAMMPS official command evidence]")
    if hits:
        lines.append("SELECTED COMMAND EVIDENCE")
        for hit in hits:
            lines.append(format_hit(hit))
    if support_hits:
        lines.append("SUPPORTING COMMON SYNTAX EVIDENCE")
        for hit in support_hits:
            lines.append(format_hit(hit))

    return {
        "query": query,
        "hits": hits,
        "support_hits": support_hits,
        "formatted_hints": "\n".join(lines).strip(),
    }


def _rank_purpose_chunks(
    chunks: Iterable[Dict[str, Any]],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    scored = _bm25_rank(
        chunks,
        query,
        fields=(
            "anchor_command",
            "family",
            "page_name",
            "title",
            "purpose_text",
            "output_context",
            "syntax",
            "examples",
            "co_occurring_commands",
            "related_commands",
            "context_text",
            "source_derivation",
        ),
        top_k=top_k,
        family_filter=PURPOSE_ALLOWED_FAMILIES,
    )
    hits: List[Dict[str, Any]] = []
    for _score_value, row in scored:
        row["command_name"] = row.get("anchor_command") or row.get("purpose_text")
        hits.append(row)
    return hits


def retrieve_lammps_purpose_hints(
    query: str,
    *,
    top_k: int = 8,
    max_chars_per_hit: int = 1400,
) -> Dict[str, Any]:
    chunks = load_lammps_purpose_chunks()
    hits = _rank_purpose_chunks(chunks, query, top_k)

    def format_hit(hit: Dict[str, Any]) -> str:
        commands = ", ".join(hit.get("co_occurring_commands") or [])
        syntax = "; ".join(hit.get("syntax") or [])
        examples = "; ".join(hit.get("examples") or [])
        body = " ".join(
            part
            for part in [
                f"purpose: {hit.get('purpose_text', '')}" if hit.get("purpose_text") else "",
                f"syntax: {syntax}" if syntax else "",
                f"output: {hit.get('output_context', '')}" if hit.get("output_context") else "",
                f"examples: {examples}" if examples else "",
                f"commands used together: {commands}" if commands else "",
                f"restrictions: {hit.get('restriction_context', '')}" if hit.get("restriction_context") else "",
            ]
            if part
        )
        return (
            f"- purpose_chunk={hit.get('chunk_id')} anchor={hit.get('anchor_command')} "
            f"[{hit.get('family')}; score={hit.get('score')}; {hit.get('source_url')}]: "
            f"{_compact(body, max_chars_per_hit)}"
        )

    lines: List[str] = []
    if hits:
        lines.append("[LAMMPS official purpose-context evidence]")
        lines.append(
            "These chunks are derived from official command-page descriptions, examples, output semantics, "
            "and related-command co-occurrence. They are not task-specific recipes."
        )
        for hit in hits:
            lines.append(format_hit(hit))

    return {
        "query": query,
        "hits": hits,
        "support_hits": [],
        "recipe_hits": [],
        "formatted_hints": "\n".join(lines).strip(),
    }


def _build_purpose_selector_candidates(
    query: str,
    *,
    lexical_top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if lexical_top_k is None:
        lexical_top_k = PURPOSE_SELECTOR_LEXICAL_TOP_K
    chunks = load_lammps_purpose_chunks()
    by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    selected: Dict[str, Dict[str, Any]] = {}

    for hit in _rank_purpose_chunks(chunks, query, lexical_top_k):
        selected[str(hit.get("chunk_id"))] = by_id.get(str(hit.get("chunk_id")), hit)

    command_rows = _rows_by_command_name(load_lammps_command_summaries())
    for chunk in chunks:
        command_name = str(chunk.get("anchor_command") or "").lower().strip()
        command_row = command_rows.get(command_name)
        if command_row and _is_state_changing_command(command_row):
            selected[str(chunk.get("chunk_id"))] = chunk

    for chunk in chunks:
        family = str(chunk.get("family"))
        if family == "howto":
            selected[str(chunk.get("chunk_id"))] = chunk

    candidates = list(selected.values())[:PURPOSE_SELECTOR_MAX_CANDIDATES]
    return candidates


def _format_purpose_candidates_for_selector(candidates: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, chunk in enumerate(candidates, start=1):
        cid = str(chunk.get("chunk_id"))
        anchor = chunk.get("anchor_command") or "(howto section)"
        commands = ", ".join(chunk.get("co_occurring_commands") or [])
        output = _compact(str(chunk.get("output_context") or ""), 180)
        restrictions = _compact(str(chunk.get("restriction_context") or ""), 120)
        context = (
            f"example_excerpt: {_compact(str(chunk.get('context_text') or ''), 240)}"
            if chunk.get("family") == "official_example"
            else ""
        )
        text = " ".join(
            part
            for part in [
                f"title: {chunk.get('title')}",
                f"purpose: {chunk.get('purpose_text')}",
                f"output: {output}" if output else "",
                f"restrictions: {restrictions}" if restrictions else "",
                f"commands: {commands}",
                context,
            ]
            if part
        )
        lines.append(
            f"[C{idx}] id={cid} family={chunk.get('family')} page={chunk.get('page_name')} "
            f"anchor={anchor}\n{_compact(text, 560 if chunk.get('family') == 'official_example' else 280)}"
        )
    return "\n\n".join(lines)


def build_lammps_purpose_selector_messages(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    max_selected: int = EVIDENCE_MAX_CORE_CHUNKS,
    max_support: int = EVIDENCE_MAX_SUPPORT_CHUNKS,
) -> List[Any]:
    from langchain.schema import HumanMessage, SystemMessage

    candidate_text = _format_purpose_candidates_for_selector(candidates)
    if len(candidate_text) > PURPOSE_SELECTOR_MAX_PROMPT_CHARS:
        candidate_text = candidate_text[:PURPOSE_SELECTOR_MAX_PROMPT_CHARS].rstrip()

    system = (
        "You are an evidence planner for LAMMPS input generation.\n"
        "Select a minimal, connected set of official command evidence for the requested Run Section objective.\n"
        "A primary chunk must define a command that produces the requested observable, supplies a required input, "
        "or consumes the result for explicit output.\n"
        "Do not select chunks merely because they share generic words such as run, output, system, atom, energy, or temperature.\n"
        "Prefer command pages over examples. Use a Howto as support for physical intent, and use an official example only "
        "when no command page or Howto provides equivalent evidence.\n"
        "Do not select velocity initialization, integration, thermostatting, barostatting, minimization, or box changes "
        "unless the user objective explicitly requires that state change.\n"
        "A request for diffusion, a trajectory, time sampling, or time averaging requires positive-length execution "
        "unless the user explicitly says to analyze an existing trajectory or current structure.\n"
        "When positive-length execution is required, include the minimum compatible protocol/integration evidence and "
        "the run command evidence when those chunks are available.\n"
        "When multiple protocol commands satisfy the same role, prefer long-established standard commands and avoid "
        "specialized integrators unless the objective specifically requires them.\n"
        "Choose output-consumer evidence when the objective requires a file, average, profile, trace, or trajectory.\n"
        "Do not write LAMMPS code. Return chunk IDs and roles only.\n"
        "Return strict JSON only."
    )
    human = "\n".join(
        [
            "USER_QUERY:",
            query,
            "",
            "CANDIDATE_OFFICIAL_EVIDENCE:",
            candidate_text,
            "",
            "Return JSON with this schema:",
            "{",
            '  "selected_chunk_ids": ["purpose::compute_rdf", "purpose::fix_ave_time"],',
            '  "support_chunk_ids": ["howto::Howto_output::0"],',
            '  "roles": {',
            '    "purpose::compute_rdf": "observable_producer",',
            '    "purpose::fix_ave_time": "output_consumer",',
            '    "howto::Howto_output::0": "guidance"',
            "  },",
            '  "protocol_change_required": false,',
            '  "rejected_reason": "Brief note on common false positives you avoided.",',
            '  "rationale": "Briefly explain why the selected chunks support the requested LAMMPS input."',
            "}",
            "",
            "Selection rules:",
            f"- Select at most {max_selected} primary command chunks.",
            f"- Select at most {max_support} supporting Howto or command chunks.",
            "- Valid roles are observable_producer, input_dependency, output_consumer, protocol, and guidance.",
            "- Every selected command must have a distinct necessary role; do not collect alternatives.",
            "- A protocol role requires an explicit state-changing objective in the user query.",
            "- A protocol selection must form an executable closure; an observable definition alone is insufficient "
            "for a requested trajectory, diffusion coefficient, time sample, or time average.",
            "- Do not select an official example when selected command pages already cover the required roles.",
            "- If a chunk is about a different physical property, reject it even if it mentions similar commands.",
            "- Never invent chunk IDs. Use only IDs present in the candidate list.",
        ]
    )
    return [
        SystemMessage(content=system),
        HumanMessage(content=human),
    ]


def _command_performs_time_integration(row: Dict[str, Any]) -> bool:
    text = _command_doc_text(row)
    return any(
        phrase in text
        for phrase in (
            "perform time integration",
            "perform plain time integration",
            "performs complete time integration",
            "updates the position and velocity",
            "update position and velocity",
        )
    )


def _command_controls_temperature(row: Dict[str, Any]) -> bool:
    text = _command_doc_text(row)
    return any(
        phrase in text
        for phrase in (
            "thermostat",
            "heat bath",
            "target temperature",
            "temperature control",
        )
    )


def _commands_conflict(
    first_name: str,
    second_name: str,
    row_map: Dict[str, Dict[str, Any]],
) -> bool:
    for owner_name, other_name in (
        (first_name, second_name),
        (second_name, first_name),
    ):
        text = _command_doc_text(row_map[owner_name])
        for match in re.finditer(
            rf"(?<![A-Za-z0-9_/]){re.escape(other_name)}(?![A-Za-z0-9_/])",
            text,
        ):
            prefix = text[max(0, match.start() - 100) : match.start()]
            if any(
                phrase in prefix
                for phrase in (
                    "do not combine",
                    "cannot be used with",
                    "must not be used with",
                    "incompatible with",
                )
            ):
                return True
    return False


def _protocol_selection_issue(
    query: str,
    selection: Dict[str, Any],
    candidate_by_id: Dict[str, Dict[str, Any]],
) -> str:
    if not _query_requires_state_change(query):
        return ""
    if not _query_requires_dynamic_execution(query):
        return ""

    row_map = _rows_by_command_name(load_lammps_command_summaries())
    selected_rows: List[Dict[str, Any]] = []
    for chunk_id in selection.get("selected_chunk_ids") or []:
        chunk = candidate_by_id.get(str(chunk_id))
        command_name = str((chunk or {}).get("anchor_command") or "").lower().strip()
        row = row_map.get(command_name)
        if row:
            selected_rows.append(row)

    missing: List[str] = []
    selected_names = {
        str(row.get("command_name") or "").lower().strip()
        for row in selected_rows
    }
    if "run" not in selected_names:
        missing.append("positive-length run command evidence")
    if not any(_command_performs_time_integration(row) for row in selected_rows):
        missing.append("a time-integration command")

    temperature_control_required = _query_requires_temperature_control(query)
    if temperature_control_required and not any(
        _command_controls_temperature(row) for row in selected_rows
    ):
        missing.append("temperature-control command evidence")

    continues_existing_state = _query_continues_existing_state(query)
    if (
        temperature_control_required
        and not continues_existing_state
        and "velocity" not in selected_names
    ):
        missing.append("velocity initialization evidence")

    conflicts: List[str] = []
    for index, first_name in enumerate(sorted(selected_names)):
        for second_name in sorted(selected_names)[index + 1 :]:
            if _commands_conflict(first_name, second_name, row_map):
                conflicts.append(f"{first_name} conflicts with {second_name}")
    if conflicts:
        missing.append("remove conflicting commands: " + "; ".join(conflicts))

    if not missing:
        return ""
    return "The dynamic execution closure is incomplete; add " + ", ".join(missing) + "."


def select_lammps_purpose_evidence(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    llm: Optional[Any] = None,
    max_selected: int = EVIDENCE_MAX_CORE_CHUNKS,
    max_support: int = EVIDENCE_MAX_SUPPORT_CHUNKS,
) -> Dict[str, Any]:
    if llm is None:
        try:
            import typing_extensions as _typing_extensions

            if not hasattr(_typing_extensions, "TypeIs") and hasattr(_typing_extensions, "TypeGuard"):
                _typing_extensions.TypeIs = _typing_extensions.TypeGuard
        except Exception:
            pass
        from config import LLM_DEFAULT

        llm = LLM_DEFAULT

    messages = build_lammps_purpose_selector_messages(
        query,
        candidates,
        max_selected=max_selected,
        max_support=max_support,
    )
    response = llm.invoke(messages)
    selection = _parse_selector_json(getattr(response, "content", "") or "")
    candidate_by_id = {str(item.get("chunk_id")): item for item in candidates}
    validation_issue_before_retry = _protocol_selection_issue(
        query,
        selection,
        candidate_by_id,
    )
    if validation_issue_before_retry:
        from langchain.schema import HumanMessage

        retry_messages = [
            *messages,
            response,
            HumanMessage(
                content=(
                    "The proposed evidence plan failed structural validation:\n"
                    f"{validation_issue_before_retry}\n"
                    "Return a corrected JSON object using only candidate chunk IDs. "
                    "Keep the observable evidence and add only the missing execution closure."
                )
            ),
        ]
        retry_response = llm.invoke(retry_messages)
        retry_selection = _parse_selector_json(
            getattr(retry_response, "content", "") or ""
        )
        if retry_selection.get("selected_chunk_ids"):
            selection = retry_selection
    validation_issue_after_retry = _protocol_selection_issue(
        query,
        selection,
        candidate_by_id,
    )
    requested_primary_ids = [
        str(x)
        for x in selection.get("selected_chunk_ids", [])
        if str(x) in candidate_by_id
    ]
    requested_support_ids = [
        str(x)
        for x in selection.get("support_chunk_ids", [])
        if str(x) in candidate_by_id and str(x) not in requested_primary_ids
    ]
    roles = selection.get("roles") if isinstance(selection.get("roles"), dict) else {}

    selected_ids: List[str] = []
    demoted_support: List[str] = []
    for chunk_id in requested_primary_ids:
        chunk = candidate_by_id[chunk_id]
        family = str(chunk.get("family") or "")
        if chunk.get("anchor_command") and family not in {"howto", "official_example"}:
            selected_ids.append(chunk_id)
        else:
            demoted_support.append(chunk_id)
        if len(selected_ids) >= max_selected:
            break

    if not selected_ids:
        for chunk in candidates:
            family = str(chunk.get("family") or "")
            if chunk.get("anchor_command") and family not in {"howto", "official_example"}:
                selected_ids.append(str(chunk.get("chunk_id")))
                break

    support_ids: List[str] = []
    for chunk_id in [*requested_support_ids, *demoted_support]:
        if chunk_id in selected_ids or chunk_id in support_ids:
            continue
        chunk = candidate_by_id[chunk_id]
        if str(chunk.get("family") or "") == "official_example" and selected_ids:
            continue
        support_ids.append(chunk_id)
        if len(support_ids) >= max_support:
            break

    role_by_id: Dict[str, str] = {}
    for chunk_id in [*selected_ids, *support_ids]:
        raw_role = str(roles.get(chunk_id) or "").strip().lower()
        if raw_role not in {
            "observable_producer",
            "input_dependency",
            "output_consumer",
            "protocol",
            "guidance",
        }:
            raw_role = "guidance" if chunk_id in support_ids else ""
        role_by_id[chunk_id] = raw_role

    selected_chunks = [candidate_by_id[x] for x in selected_ids + support_ids]
    return {
        "selection": selection,
        "selected_ids": selected_ids,
        "support_ids": support_ids,
        "role_by_id": role_by_id,
        "protocol_change_required": bool(selection.get("protocol_change_required")),
        "validation_issue_before_retry": validation_issue_before_retry,
        "validation_issue_after_retry": validation_issue_after_retry,
        "selected_chunks": selected_chunks,
    }


def retrieve_lammps_llm_selected_purpose_hints(
    query: str,
    *,
    llm: Optional[Any] = None,
    lexical_top_k: Optional[int] = None,
    max_selected: int = EVIDENCE_MAX_CORE_CHUNKS,
    max_support: int = EVIDENCE_MAX_SUPPORT_CHUNKS,
    max_chars_per_hit: int = 1700,
) -> Dict[str, Any]:
    candidates = _build_purpose_selector_candidates(query, lexical_top_k=lexical_top_k)
    try:
        selected = select_lammps_purpose_evidence(
            query,
            candidates,
            llm=llm,
            max_selected=max_selected,
            max_support=max_support,
        )
        hits = selected.get("selected_chunks") or []
        selector_error = ""
    except Exception as exc:
        selected = {
            "selection": {},
            "selected_ids": [],
            "support_ids": [],
            "role_by_id": {},
            "protocol_change_required": False,
            "validation_issue_before_retry": "",
            "validation_issue_after_retry": "",
            "selected_chunks": [],
        }
        ranked = _rank_purpose_chunks(load_lammps_purpose_chunks(), query, max_selected + max_support)
        hits = [
            hit
            for hit in ranked
            if hit.get("anchor_command")
            and str(hit.get("family") or "") not in {"howto", "official_example"}
        ][:max_selected]
        selector_error = repr(exc)

    def format_hit(hit: Dict[str, Any]) -> str:
        commands = ", ".join(hit.get("co_occurring_commands") or [])
        syntax = "; ".join(hit.get("syntax") or [])
        examples = "; ".join(hit.get("examples") or [])
        body = " ".join(
            part
            for part in [
                f"purpose: {hit.get('purpose_text', '')}" if hit.get("purpose_text") else "",
                f"syntax: {syntax}" if syntax else "",
                f"output: {hit.get('output_context', '')}" if hit.get("output_context") else "",
                f"examples: {examples}" if examples else "",
                f"commands used together: {commands}" if commands else "",
                f"context: {hit.get('context_text', '')}" if hit.get("context_text") else "",
            ]
            if part
        )
        return (
            f"- [{hit.get('chunk_id')}] anchor={hit.get('anchor_command') or '(howto section)'} "
            f"[{hit.get('family')}; {hit.get('source_url')}]: {_compact(body, max_chars_per_hit)}"
        )

    lines = [
        "[LLM-selected LAMMPS official purpose-context evidence]",
        "Candidate chunks were derived from official LAMMPS command/Howto pages; the selector only chose chunk IDs.",
    ]
    for hit in hits:
        lines.append(format_hit(hit))
    rationale = (selected.get("selection") or {}).get("rationale")
    rejected = (selected.get("selection") or {}).get("rejected_reason")
    if rationale:
        lines.extend(["", "[Evidence selector rationale]", _compact(str(rationale), 1200)])
    if rejected:
        lines.extend(["", "[Rejected false positives]", _compact(str(rejected), 800)])
    if selector_error:
        lines.extend(["", "[Evidence selector fallback]", selector_error])

    return {
        "query": query,
        "candidate_count": len(candidates),
        "hits": hits,
        "support_hits": [],
        "recipe_hits": [],
        "selected_ids": selected.get("selected_ids", []),
        "support_ids": selected.get("support_ids", []),
        "role_by_id": selected.get("role_by_id", {}),
        "protocol_change_required": bool(selected.get("protocol_change_required")),
        "selector_validation_issue_before_retry": selected.get(
            "validation_issue_before_retry",
            "",
        ),
        "selector_validation_issue_after_retry": selected.get(
            "validation_issue_after_retry",
            "",
        ),
        "selector_selection": selected.get("selection", {}),
        "selector_error": selector_error,
        "formatted_hints": "\n".join(lines).strip(),
    }


def _command_doc_text(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(field) or "")
        for field in (
            "command_name",
            "short_summary",
            "output_info_summary",
            "restrictions_summary",
            "syntax",
        )
    ).lower()


def _output_shapes(row: Dict[str, Any]) -> set:
    text = str(row.get("output_info_summary") or "").lower()
    shapes = set()
    for shape, phrase in (
        ("global_array", "global array"),
        ("global_vector", "global vector"),
        ("global_scalar", "global scalar"),
        ("per_atom_array", "per-atom array"),
        ("per_atom_vector", "per-atom vector"),
        ("local_array", "local array"),
        ("local_vector", "local vector"),
    ):
        if phrase in text:
            shapes.add(shape)
    return shapes


def _accepted_input_shapes(row: Dict[str, Any]) -> set:
    text = " ".join(
        [
            str(row.get("short_summary") or ""),
            str(row.get("syntax") or ""),
        ]
    ).lower()
    shapes = set()
    for shape, phrase in (
        ("global_array", "global array"),
        ("global_vector", "global vector"),
        ("global_scalar", "global scalar"),
        ("per_atom_array", "per-atom array"),
        ("per_atom_vector", "per-atom vector"),
        ("local_array", "local array"),
        ("local_vector", "local vector"),
    ):
        if phrase in text:
            shapes.add(shape)
    return shapes


def _writes_property_output(row: Dict[str, Any]) -> bool:
    text = _command_doc_text(row)
    return (
        str(row.get("family") or "") == "dump"
        or "written to a file" in text
        or "write to a file" in text
        or "file output" in text
        or "print thermodynamic" in text
        or "begins logging information" in text
        or "printing thermodynamic data" in text
    )


def _is_state_changing_command(row: Dict[str, Any]) -> bool:
    text = _command_doc_text(row)
    command_name = str(row.get("command_name") or "").lower().strip()
    family = str(row.get("family") or "").lower().strip()
    if command_name == "run":
        return True
    if family == "fix" and any(
        phrase in text
        for phrase in (
            "time integration",
            "heat bath",
            "thermostat",
            "barostat",
            "updates the position and velocity",
            "update position and velocity",
        )
    ):
        return True
    return any(
        phrase in text
        for phrase in (
            "perform time integration",
            "perform plain time integration",
            "performs complete time integration",
            "run or continue dynamics",
            "updates the position and velocity",
            "set or change the velocities",
            "perform an energy minimization",
            "adjusting atom coordinates",
            "simulation box during an energy minimization",
            "change the volume and/or shape",
            "displace a group of atoms",
        )
    )


def _query_requires_state_change(query: str) -> bool:
    passive_source = bool(
        re.search(
            r"\b(?:existing|stored|precomputed|previously generated)\b"
            r"[^.!?]{0,120}\btrajectory\b|"
            r"\bcurrent\s+structure\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    explicit_execution = bool(
        re.search(
            r"\b(?:run|perform|simulate|equilibrat\w*|propagat\w*)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    if passive_source and not explicit_execution:
        return False
    return bool(
        re.search(
            r"\b(?:molecular\s+dynamics|dynamics|trajectory|equilibrat\w*|"
            r"diffus\w*|nvt|npt|nve|nph|thermostat\w*|barostat\w*|"
            r"relax\w*|minimi[sz]\w*|strain\w*|deform\w*|thermal\s+expansion|"
            r"time[- ]averag\w*|sampled|sampling)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )


def _query_requires_dynamic_execution(query: str) -> bool:
    return bool(
        _query_requires_state_change(query)
        and re.search(
            r"\b(?:molecular\s+dynamics|dynamics|trajectory|diffus\w*|"
            r"nvt|npt|nve|nph|time[- ]averag\w*|sampled|sampling)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )


def _query_requires_temperature_control(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:constant[- ]temperature|nvt|npt|thermostat\w*)\b|"
            r"\b\d+(?:\.\d+)?\s*K\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )


def _query_continues_existing_state(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:continue|restart|existing|stored|precomputed)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )


def _query_requests_partitioned_output(query: str) -> bool:
    text = query or ""
    if re.search(
        r"\b(?:per[- ](?:atom|molecule|species|group|chunk|bin|layer)|"
        r"each\s+(?:atom|molecule|species|group)|"
        r"molecules|"
        r"spatial|profile|distribution|grid|voxel|bin(?:ned|ning)?|layer)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(r"\bmolecular\b", text, flags=re.IGNORECASE)
        and not re.search(r"\bmolecular\s+dynamics\b", text, flags=re.IGNORECASE)
    )


def _command_variant_family(command_name: str) -> str:
    parts = str(command_name or "").lower().split()
    if len(parts) < 2:
        return str(command_name or "").lower()
    return f"{parts[0]} {parts[1].split('/', 1)[0]}"


def _producer_granularity_score(query: str, row: Dict[str, Any]) -> float:
    text = _command_doc_text(row)
    partitioned_doc = any(
        phrase in text
        for phrase in (
            "multiple chunks",
            "per-atom",
            "per atom",
            "per-grid",
            "per grid",
            "each chunk",
            "each molecule",
            "spatial",
        )
    )
    if _query_requests_partitioned_output(query):
        return 5.0 if partitioned_doc else -2.0
    return 0.0


def _best_observable_variant(
    query: str,
    command_name: str,
    row_map: Dict[str, Dict[str, Any]],
) -> str:
    family = _command_variant_family(command_name)
    candidates: List[Tuple[float, str]] = []
    for name, row in row_map.items():
        if (
            _command_variant_family(name) != family
            or _infer_command_role(row) != "observable_producer"
        ):
            continue
        score = _score(_tokens(query), _command_doc_text(row))
        score += _producer_granularity_score(query, row)
        candidates.append((score, name))
    if not candidates:
        return command_name
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _command_style_is_explicit(query: str, command_name: str) -> bool:
    style = command_name.split(" ", 1)[-1]
    style_parts = [
        part
        for part in re.split(r"[/_-]+", style)
        if len(part) > 1
    ]
    query_lower = (query or "").lower()
    explicit_parts = {
        part
        for part in style_parts
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(part)}(?![A-Za-z0-9])",
            query_lower,
        )
    }
    if "/" in style and style_parts[1:]:
        return any(part in explicit_parts for part in style_parts[1:])
    return bool(explicit_parts)


def _best_standard_protocol_command(
    query: str,
    row_map: Dict[str, Dict[str, Any]],
    *,
    require_integration: bool,
    require_temperature_control: bool,
) -> Optional[str]:
    query_tokens = _tokens(query)
    candidates: List[Tuple[float, int, str]] = []
    for name, row in row_map.items():
        if str(row.get("family") or "") != "fix":
            continue
        integrates = _command_performs_time_integration(row)
        controls_temperature = _command_controls_temperature(row)
        if require_integration and not integrates:
            continue
        if require_temperature_control and not controls_temperature:
            continue
        text = _command_doc_text(row)
        style = name.split(" ", 1)[-1]
        style_is_explicit = _command_style_is_explicit(query, name)
        mismatch_penalty = 0.0
        if "electron force field" in text and not re.search(
            r"\b(?:electron|electronic)\b",
            query or "",
            flags=re.IGNORECASE,
        ):
            mismatch_penalty += 8.0
        if (
            ("npt" in style or "nph" in style)
            and ("barostat" in text or "box dimensions" in text)
            and not re.search(
            r"\b(?:pressure|npt|barostat|cell|box|volume)\b",
            query or "",
            flags=re.IGNORECASE,
            )
        ):
            mismatch_penalty += 6.0
        if "isokinetic" in text and not re.search(
            r"\b(?:isokinetic|kinetic energy)\b",
            query or "",
            flags=re.IGNORECASE,
        ):
            mismatch_penalty += 6.0
        specialization_penalty = 3 if "/" in style else 0
        relevance = (
            10.0 * _score(query_tokens, text) if style_is_explicit else 0.0
        ) - mismatch_penalty
        candidates.append((relevance, -specialization_penalty, name))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            len(item[2]),
            item[2],
        )
    )
    return candidates[0][2]


def _infer_command_role(row: Dict[str, Any]) -> str:
    family = str(row.get("family") or "")
    if _is_state_changing_command(row):
        return "protocol"
    if family == "compute":
        return "observable_producer"
    if _writes_property_output(row):
        return "output_consumer"
    return "input_dependency"


def _validated_command_role(
    row: Dict[str, Any],
    requested_role: str,
) -> str:
    inferred = _infer_command_role(row)
    if inferred == "protocol":
        return inferred
    if requested_role == "observable_producer" and not _output_shapes(row):
        return inferred
    if requested_role == "output_consumer" and inferred != "output_consumer":
        return inferred
    if requested_role in {
        "observable_producer",
        "input_dependency",
        "output_consumer",
    }:
        return requested_role
    return inferred


def _dependency_is_connected(
    dependency_name: str,
    producer_names: Iterable[str],
    row_map: Dict[str, Dict[str, Any]],
) -> bool:
    dependency = row_map[dependency_name]
    dependency_style = dependency_name.split(" ", 1)[-1]
    dependency_stem = dependency_style.split("/", 1)[0]
    dependency_shapes = _output_shapes(dependency)
    for producer_name in producer_names:
        producer = row_map[producer_name]
        if dependency_name in _document_referenced_commands(producer, row_map):
            return True
        related = {
            str(value or "").lower().strip()
            for value in producer.get("related_commands") or []
        }
        if dependency_name in related:
            return True
        syntax = " ".join(str(value or "") for value in producer.get("syntax") or [])
        if (
            dependency_shapes
            and re.search(
                rf"(?<![A-Za-z0-9]){re.escape(dependency_stem)}\s*-\s*ID(?![A-Za-z0-9])",
                syntax,
                flags=re.IGNORECASE,
            )
        ):
            return True
    return False


def _output_consumer_matches_query(row: Dict[str, Any], query: str) -> bool:
    text = _command_doc_text(row)
    needs_average = bool(
        re.search(
            r"\b(?:average|averaged|averaging|mean)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    needs_file = bool(
        re.search(
            r"\b(?:file|write|save|record|output)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    if needs_average and "averag" not in text:
        return False
    if needs_file and not _writes_property_output(row):
        return False
    return True


def _shapes_are_compatible(produced: set, accepted: set) -> bool:
    if produced & accepted:
        return True
    if "global_vector" in produced and "global_scalar" in accepted:
        return True
    if "per_atom_array" in produced and "per_atom_vector" in accepted:
        return True
    return False


def _document_referenced_commands(
    row: Dict[str, Any],
    row_map: Dict[str, Dict[str, Any]],
) -> List[str]:
    text = _command_doc_text(row)
    own_name = str(row.get("command_name") or "").lower().strip()
    references: List[str] = []
    for name in sorted(row_map, key=len, reverse=True):
        candidate = row_map[name]
        if (
            name == own_name
            or " " not in name
            or name.endswith(" styles")
            or str(candidate.get("family") or "").endswith("_style")
        ):
            continue
        matches = list(
            re.finditer(
            rf"(?<![A-Za-z0-9_/]){re.escape(name)}(?![A-Za-z0-9_/])",
            text,
            )
        )
        for match in matches:
            window = text[
                max(0, match.start() - 120) : min(len(text), match.end() + 120)
            ]
            if any(
                phrase in window
                for phrase in (
                    "defined by",
                    "specified by",
                    "requires",
                    "required",
                    "calculated by",
                    "provided by",
                    "input to",
                    "input is",
                )
            ) and not any(
                phrase in window
                for phrase in (
                    "for example",
                    "see the",
                    "see ",
                    "cannot be used",
                )
            ):
                references.append(name)
                break
    return references


def _syntax_referenced_compute_commands(
    row: Dict[str, Any],
    row_map: Dict[str, Dict[str, Any]],
) -> List[str]:
    syntax = " ".join(str(value or "") for value in row.get("syntax") or [])
    stems = [
        stem.lower()
        for stem in re.findall(
            r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9_]*)\s*-\s*ID(?![A-Za-z0-9])",
            syntax,
            flags=re.IGNORECASE,
        )
        if stem.lower() not in {"group"}
    ]
    producer_text = _command_doc_text(row)
    atom_level_input = any(
        phrase in producer_text
        for phrase in (
            "contributions from atoms",
            "per-atom inputs",
            "per-atom values",
        )
    )
    chunk_level_input = "chunks are collections of atoms" in producer_text
    references: List[str] = []
    for stem in dict.fromkeys(stems):
        candidates: List[Tuple[float, str]] = []
        for name, candidate in row_map.items():
            if not name.startswith(f"compute {stem}"):
                continue
            style = name.split(" ", 1)[1]
            if style != stem and not style.startswith(f"{stem}/"):
                continue
            score = _score(_tokens(producer_text), _command_doc_text(candidate))
            shapes = _output_shapes(candidate)
            if atom_level_input and shapes & {"per_atom_vector", "per_atom_array"}:
                score += 8.0
            if chunk_level_input and "chunk" in style:
                score += 8.0
            candidates.append((score, name))
        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            references.append(candidates[0][1])
    return references


def _best_output_consumer(
    producer: Dict[str, Any],
    query: str,
    row_map: Dict[str, Dict[str, Any]],
    excluded: set,
) -> Optional[str]:
    produced = _output_shapes(producer)
    if not produced:
        return None
    related = {
        str(value or "").lower().strip()
        for value in producer.get("related_commands") or []
    }
    query_tokens = _tokens(query)
    query_requests_correlation = bool(
        re.search(
            r"\b(?:correlat\w*|autocorrelat\w*|green[- ]kubo)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    producer_already_is_correlation = "correlation function" in _command_doc_text(
        producer
    )
    needs_correlation_consumer = (
        query_requests_correlation and not producer_already_is_correlation
    )
    scored: List[Tuple[float, str]] = []
    for name, candidate in row_map.items():
        if name in excluded or _is_state_changing_command(candidate):
            continue
        accepted = _accepted_input_shapes(candidate)
        if not accepted or not _shapes_are_compatible(produced, accepted):
            continue
        if not _writes_property_output(candidate):
            continue
        score = _score(query_tokens, _command_doc_text(candidate))
        if name in related:
            score += 6.0
        candidate_text = _command_doc_text(candidate)
        candidate_is_correlation = "correlation" in candidate_text
        if needs_correlation_consumer and candidate_is_correlation:
            score += 4.0
        elif candidate_is_correlation:
            score -= 8.0
        if re.search(r"\b(?:average|averaged|mean|sampled)\b", query, re.IGNORECASE):
            if "averag" in candidate_text:
                score += 2.0
        if re.search(r"\b(?:file|write|save|record|output)\b", query, re.IGNORECASE):
            if "file" in candidate_text:
                score += 2.0
        scored.append((score, name))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _best_query_output_command(
    query: str,
    row_map: Dict[str, Dict[str, Any]],
    excluded: set,
) -> Optional[str]:
    needs_average = bool(
        re.search(
            r"\b(?:average|averaged|averaging|mean)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    needs_file = bool(
        re.search(
            r"\b(?:file|write|save|record|output)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    if not needs_average and not needs_file:
        return None
    partitioned_output = _query_requests_partitioned_output(query)
    query_tokens = _tokens(query)
    scored: List[Tuple[float, str]] = []
    for name, candidate in row_map.items():
        if (
            name in excluded
            or _is_state_changing_command(candidate)
            or not _writes_property_output(candidate)
        ):
            continue
        text = _command_doc_text(candidate)
        accepted = _accepted_input_shapes(candidate)
        accepts_global = bool(
            accepted & {"global_scalar", "global_vector", "global_array"}
        )
        accepts_partitioned = bool(
            accepted
            & {
                "per_atom_vector",
                "per_atom_array",
                "local_vector",
                "local_array",
            }
        )
        if not partitioned_output and accepted and not accepts_global:
            continue
        if needs_average and "averag" not in text:
            continue
        score = _score(query_tokens, text)
        if partitioned_output and accepts_partitioned:
            score += 3.0
        elif not partitioned_output and accepts_global:
            score += 3.0
        if needs_average:
            score += 4.0
        if needs_file and "file" in text:
            score += 2.0
        if "correlation" in text and not re.search(
            r"\b(?:correlat\w*|autocorrelat\w*|green[- ]kubo)\b",
            query or "",
            flags=re.IGNORECASE,
        ):
            score -= 8.0
        scored.append((score, name))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def build_lammps_command_evidence_plan(
    query: str,
    purpose_hits: Iterable[Dict[str, Any]],
    *,
    role_by_id: Optional[Dict[str, str]] = None,
    max_core_commands: int = EVIDENCE_MAX_CORE_CHUNKS,
    max_dependency_commands: int = EVIDENCE_MAX_DEPENDENCY_COMMANDS,
) -> Dict[str, Any]:
    rows = load_lammps_command_summaries()
    row_map = _rows_by_command_name(rows)
    roles = role_by_id or {}
    protocol_allowed = _query_requires_state_change(query)
    hits = list(purpose_hits)
    core_candidates: List[Dict[str, Any]] = []
    for hit in hits:
        name = str(hit.get("anchor_command") or "").lower().strip()
        row = row_map.get(name)
        if not row or any(
            item["command_name"] == name for item in core_candidates
        ):
            continue
        chunk_id = str(hit.get("chunk_id") or "")
        requested_role = str(roles.get(chunk_id) or "").strip().lower()
        role = _validated_command_role(row, requested_role)
        if role == "protocol" and not protocol_allowed:
            continue
        relevance_text = " ".join(
            [
                _command_doc_text(row),
                str(hit.get("purpose_text") or ""),
                str(hit.get("output_context") or ""),
            ]
        )
        core_candidates.append(
            {
                "command_name": name,
                "role": role,
                "source_chunk_id": chunk_id,
                "reason": "selected_by_evidence_planner",
                "_relevance": _score(_tokens(query), relevance_text),
            }
        )

    normalized_candidates: List[Dict[str, Any]] = []
    seen_normalized = set()
    for item in core_candidates:
        normalized = dict(item)
        if item["role"] == "observable_producer":
            variant = _best_observable_variant(
                query,
                item["command_name"],
                row_map,
            )
            if variant != item["command_name"]:
                normalized["command_name"] = variant
                normalized["source_chunk_id"] = ""
                normalized["reason"] = (
                    f"official_variant_match:{item['command_name']}"
                )
                normalized["_relevance"] = _score(
                    _tokens(query),
                    _command_doc_text(row_map[variant]),
                )
        key = (normalized["command_name"], normalized["role"])
        if key not in seen_normalized:
            normalized_candidates.append(normalized)
            seen_normalized.add(key)
    core_candidates = normalized_candidates

    conflict_filtered: List[Dict[str, Any]] = []
    for item in core_candidates:
        conflicting_index = next(
            (
                index
                for index, retained in enumerate(conflict_filtered)
                if _commands_conflict(
                    item["command_name"],
                    retained["command_name"],
                    row_map,
                )
            ),
            None,
        )
        if conflicting_index is None:
            conflict_filtered.append(item)
            continue
        retained = conflict_filtered[conflicting_index]

        def protocol_capability_count(command_name: str) -> int:
            row = row_map[command_name]
            return int(_command_performs_time_integration(row)) + int(
                _command_controls_temperature(row)
            )

        if protocol_capability_count(item["command_name"]) > protocol_capability_count(
            retained["command_name"]
        ):
            conflict_filtered[conflicting_index] = item
    core_candidates = conflict_filtered

    if _query_requires_dynamic_execution(query):
        temperature_required = _query_requires_temperature_control(query)
        selected_method_protocols = [
            item
            for item in core_candidates
            if item["role"] == "protocol"
            and (
                _command_performs_time_integration(
                    row_map[item["command_name"]]
                )
                or _command_controls_temperature(
                    row_map[item["command_name"]]
                )
            )
        ]
        if (
            temperature_required
            and selected_method_protocols
            and not any(
                _command_style_is_explicit(query, item["command_name"])
                for item in selected_method_protocols
            )
        ):
            preferred_standard = _best_standard_protocol_command(
                query,
                row_map,
                require_integration=True,
                require_temperature_control=True,
            )
            core_candidates = [
                item
                for item in core_candidates
                if item not in selected_method_protocols
            ]
            if preferred_standard:
                core_candidates.append(
                    {
                        "command_name": preferred_standard,
                        "role": "protocol",
                        "source_chunk_id": "",
                        "reason": "generic_standard_protocol",
                        "_relevance": 0.0,
                    }
                )

        combined_protocols = [
            item
            for item in core_candidates
            if item["role"] == "protocol"
            and _command_performs_time_integration(
                row_map[item["command_name"]]
            )
            and _command_controls_temperature(
                row_map[item["command_name"]]
            )
        ]
        if temperature_required and combined_protocols:
            preferred_combined = _best_standard_protocol_command(
                query,
                row_map,
                require_integration=True,
                require_temperature_control=True,
            )
            core_candidates = [
                item
                for item in core_candidates
                if not (
                    item["role"] == "protocol"
                    and (
                        _command_performs_time_integration(
                            row_map[item["command_name"]]
                        )
                        or _command_controls_temperature(
                            row_map[item["command_name"]]
                        )
                    )
                )
            ]
            if preferred_combined:
                core_candidates.append(
                    {
                        "command_name": preferred_combined,
                        "role": "protocol",
                        "source_chunk_id": "",
                        "reason": "generic_protocol_consolidation",
                        "_relevance": 0.0,
                    }
                )

        protocol_rows = [
            row_map[item["command_name"]]
            for item in core_candidates
            if item["role"] == "protocol"
        ]
        has_integration = any(
            _command_performs_time_integration(row) for row in protocol_rows
        )
        has_temperature_control = any(
            _command_controls_temperature(row) for row in protocol_rows
        )
        if not has_integration or (
            temperature_required and not has_temperature_control
        ):
            replacement = _best_standard_protocol_command(
                query,
                row_map,
                require_integration=True,
                require_temperature_control=temperature_required,
            )
            if replacement:
                core_candidates = [
                    item
                    for item in core_candidates
                    if not (
                        item["role"] == "protocol"
                        and (
                            _command_performs_time_integration(
                                row_map[item["command_name"]]
                            )
                            or _command_controls_temperature(
                                row_map[item["command_name"]]
                            )
                        )
                    )
                ]
                if not any(
                    item["command_name"] == replacement
                    for item in core_candidates
                ):
                    core_candidates.append(
                        {
                            "command_name": replacement,
                            "role": "protocol",
                            "source_chunk_id": "",
                            "reason": "generic_execution_closure",
                            "_relevance": 0.0,
                        }
                    )

        closure_commands = ["run"]
        if (
            temperature_required
            and not _query_continues_existing_state(query)
        ):
            closure_commands.append("velocity")
        for command_name in closure_commands:
            if command_name in row_map and not any(
                item["command_name"] == command_name
                for item in core_candidates
            ):
                core_candidates.append(
                    {
                        "command_name": command_name,
                        "role": "protocol",
                        "source_chunk_id": "",
                        "reason": "generic_execution_closure",
                        "_relevance": 0.0,
                    }
                )

    role_priority = {
        "observable_producer": 0,
        "protocol": 1,
        "input_dependency": 2,
        "output_consumer": 3,
    }
    core_candidates.sort(
        key=lambda item: role_priority.get(item["role"], 4)
    )

    producer_variant_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in core_candidates:
        if item["role"] != "observable_producer":
            continue
        family = _command_variant_family(item["command_name"])
        producer_variant_groups.setdefault(family, []).append(item)

    retained_producer_ids = set()
    for candidates in producer_variant_groups.values():
        best = max(
            candidates,
            key=lambda item: (
                item["_relevance"]
                + _producer_granularity_score(
                    query,
                    row_map[item["command_name"]],
                )
            ),
        )
        retained_producer_ids.add(id(best))

    retained_producer_names = [
        item["command_name"]
        for item in core_candidates
        if id(item) in retained_producer_ids
    ]
    core: List[Dict[str, Any]] = []
    for item in core_candidates:
        if (
            item["role"] == "observable_producer"
            and id(item) not in retained_producer_ids
        ):
            continue
        if item["role"] == "input_dependency" and not _dependency_is_connected(
            item["command_name"],
            retained_producer_names,
            row_map,
        ):
            continue
        if item["role"] == "output_consumer":
            row = row_map[item["command_name"]]
            compatible = any(
                _shapes_are_compatible(
                    _output_shapes(row_map[producer_name]),
                    _accepted_input_shapes(row),
                )
                for producer_name in retained_producer_names
            )
            if retained_producer_names and not compatible:
                continue
            if not _output_consumer_matches_query(row, query):
                continue
        clean_item = {
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        }
        row = row_map[clean_item["command_name"]]
        clean_item["produces"] = sorted(_output_shapes(row))
        clean_item["accepts"] = sorted(_accepted_input_shapes(row))
        core.append(clean_item)
        if len(core) >= max_core_commands:
            break
    core_names = [item["command_name"] for item in core]

    dependencies: List[Dict[str, Any]] = []
    dependency_names: List[str] = []

    def add_dependency(name: str, role: str, reason: str) -> None:
        if (
            name not in row_map
            or name in core_names
            or name in dependency_names
            or len(dependencies) >= max_dependency_commands
        ):
            return
        row = row_map[name]
        if _is_state_changing_command(row) and not protocol_allowed:
            return
        dependency_names.append(name)
        dependencies.append(
            {
                "command_name": name,
                "role": role,
                "source_chunk_id": "",
                "reason": reason,
                "produces": sorted(_output_shapes(row)),
                "accepts": sorted(_accepted_input_shapes(row)),
            }
        )

    for item in core:
        row = row_map[item["command_name"]]
        referenced_commands = _document_referenced_commands(row, row_map)
        if item["role"] == "observable_producer":
            referenced_commands.extend(
                _syntax_referenced_compute_commands(row, row_map)
            )
        for name in dict.fromkeys(referenced_commands):
            referenced = row_map[name]
            role = _infer_command_role(referenced)
            if role in {"observable_producer", "input_dependency"}:
                add_dependency(
                    name,
                    "input_dependency",
                    f"referenced_by_official_doc:{item['command_name']}",
                )

    existing = set(core_names) | set(dependency_names)
    for item in core:
        if item["role"] != "observable_producer":
            continue
        producer = row_map[item["command_name"]]
        consumer = _best_output_consumer(
            producer,
            query,
            row_map,
            existing,
        )
        if consumer:
            add_dependency(
                consumer,
                "output_consumer",
                f"output_shape_match:{item['command_name']}",
            )
            existing.add(consumer)

    existing = set(core_names) | set(dependency_names)
    selected_output_rows = [
        row_map[item["command_name"]]
        for item in [*core, *dependencies]
        if item["role"] == "output_consumer"
    ]
    needs_average = bool(
        re.search(
            r"\b(?:average|averaged|averaging|mean)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    has_matching_output = bool(selected_output_rows) and (
        not needs_average
        or any("averag" in _command_doc_text(row) for row in selected_output_rows)
    )
    if not has_matching_output:
        output_command = _best_query_output_command(
            query,
            row_map,
            existing,
        )
        if output_command:
            add_dependency(
                output_command,
                "output_consumer",
                "query_output_requirement",
            )

    return {
        "query": query,
        "core": core,
        "dependencies": dependencies,
        "command_names": [*core_names, *dependency_names],
        "protocol_changes_allowed": protocol_allowed,
        "uses_property_command_map": False,
    }


def build_selector_preserved_evidence_plan(
    query: str,
    purpose_hits: Iterable[Dict[str, Any]],
    *,
    selected_ids: Iterable[str],
    role_by_id: Optional[Dict[str, str]] = None,
    max_core_commands: int = EVIDENCE_MAX_CORE_CHUNKS,
    max_dependency_commands: int = EVIDENCE_MAX_DEPENDENCY_COMMANDS,
) -> Dict[str, Any]:
    rows = load_lammps_command_summaries()
    row_map = _rows_by_command_name(rows)
    selected = set(str(value) for value in selected_ids)
    roles = role_by_id or {}
    core: List[Dict[str, Any]] = []
    for hit in purpose_hits:
        chunk_id = str(hit.get("chunk_id") or "")
        if chunk_id not in selected:
            continue
        name = str(hit.get("anchor_command") or "").lower().strip()
        row = row_map.get(name)
        if not row or any(item["command_name"] == name for item in core):
            continue
        requested_role = str(roles.get(chunk_id) or "").strip().lower()
        role = _validated_command_role(row, requested_role)
        core.append(
            {
                "command_name": name,
                "role": role,
                "source_chunk_id": chunk_id,
                "reason": "preserved_llm_selector_decision",
                "produces": sorted(_output_shapes(row)),
                "accepts": sorted(_accepted_input_shapes(row)),
            }
        )
        if len(core) >= max_core_commands:
            break

    core_names = [item["command_name"] for item in core]
    dependencies: List[Dict[str, Any]] = []
    dependency_names = set()

    def add_dependency(name: str, reason: str) -> None:
        if (
            name not in row_map
            or name in core_names
            or name in dependency_names
            or len(dependencies) >= max_dependency_commands
        ):
            return
        row = row_map[name]
        role = _infer_command_role(row)
        if role == "protocol":
            return
        dependency_names.add(name)
        dependencies.append(
            {
                "command_name": name,
                "role": (
                    role
                    if role in {"input_dependency", "output_consumer"}
                    else "input_dependency"
                ),
                "source_chunk_id": "",
                "reason": reason,
                "produces": sorted(_output_shapes(row)),
                "accepts": sorted(_accepted_input_shapes(row)),
            }
        )

    for item in core:
        row = row_map[item["command_name"]]
        references = [
            *_document_referenced_commands(row, row_map),
            *_syntax_referenced_compute_commands(row, row_map),
        ]
        for name in dict.fromkeys(references):
            add_dependency(
                name,
                f"documented_dependency_of:{item['command_name']}",
            )

    selected_output_rows = [
        row_map[item["command_name"]]
        for item in core
        if item["role"] == "output_consumer"
    ]
    needs_average = bool(
        re.search(
            r"\b(?:average|averaged|averaging|mean)\b",
            query or "",
            flags=re.IGNORECASE,
        )
    )
    selected_output_satisfies_request = bool(selected_output_rows) and (
        not needs_average
        or any(
            "averag" in _command_doc_text(row)
            for row in selected_output_rows
        )
    )
    if not selected_output_satisfies_request:
        producer_names = [
            item["command_name"]
            for item in core
            if item["role"] == "observable_producer"
        ]
        for producer_name in producer_names:
            consumer = _best_output_consumer(
                row_map[producer_name],
                query,
                row_map,
                set(core_names) | dependency_names,
            )
            if consumer:
                add_dependency(
                    consumer,
                    f"documented_shape_consumer_of:{producer_name}",
                )

    return {
        "query": query,
        "core": core,
        "dependencies": dependencies,
        "command_names": [
            *core_names,
            *[item["command_name"] for item in dependencies],
        ],
        "protocol_changes_allowed": bool(
            _query_requires_state_change(query)
            and any(item["role"] == "protocol" for item in core)
        ),
        "uses_property_command_map": False,
        "selector_preserved": True,
    }


def _format_command_evidence_plan(
    plan: Dict[str, Any],
    purpose_info: Dict[str, Any],
) -> str:
    lines = [
        "[LAMMPS EVIDENCE PLAN]",
        "ADVISORY STATUS: candidate commands only; none is mandatory.",
        "The candidates below were retrieved from official documentation and completed by generic dependency matching.",
        "They are not a property-specific recipe and must not override the structured calculation intent.",
        "CORE CANDIDATES",
    ]
    core = plan.get("core") or []
    dependencies = plan.get("dependencies") or []
    if core:
        for item in core:
            shape_details = " ".join(
                detail
                for detail in (
                    f"produces={','.join(item.get('produces') or [])}"
                    if item.get("produces")
                    else "",
                    f"accepts={','.join(item.get('accepts') or [])}"
                    if item.get("accepts")
                    else "",
                )
                if detail
            )
            lines.append(
                f"- {item.get('command_name')} role={item.get('role')} "
                f"source={item.get('source_chunk_id') or 'official_command_doc'} "
                f"{shape_details}".rstrip()
            )
    else:
        lines.append("- (none)")
    lines.append("CANDIDATE DEPENDENCIES")
    if dependencies:
        for item in dependencies:
            shape_details = " ".join(
                detail
                for detail in (
                    f"produces={','.join(item.get('produces') or [])}"
                    if item.get("produces")
                    else "",
                    f"accepts={','.join(item.get('accepts') or [])}"
                    if item.get("accepts")
                    else "",
                )
                if detail
            )
            lines.append(
                f"- {item.get('command_name')} role={item.get('role')} "
                f"reason={item.get('reason')} {shape_details}".rstrip()
            )
    else:
        lines.append("- (none)")

    selected_ids = set(purpose_info.get("selected_ids") or [])
    selected_hits = [
        hit
        for hit in purpose_info.get("hits") or []
        if str(hit.get("chunk_id") or "") in selected_ids
    ]
    lines.append("SELECTOR-PRESERVED PRIMARY EVIDENCE")
    if selected_hits:
        for hit in selected_hits:
            lines.append(
                f"- [{hit.get('chunk_id')}] "
                f"anchor={hit.get('anchor_command') or '(guidance)'} "
                f"purpose={_compact(str(hit.get('purpose_text') or ''), 500)}"
            )
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "CANDIDATE POLICY",
            "- Use the structured IntentSpec, not this retrieval result, to decide whether state change is required.",
            "- Ignore any candidate that is unnecessary, incompatible, or less suitable than the baseline command.",
            "- A specialized command must not replace an executable standard command without an intent-driven reason.",
        ]
    )

    support_ids = set(purpose_info.get("support_ids") or [])
    support_hits = [
        hit
        for hit in purpose_info.get("hits") or []
        if str(hit.get("chunk_id") or "") in support_ids
        and str(hit.get("family") or "") != "official_example"
    ]
    if support_hits:
        lines.append("SUPPORTING PURPOSE CONTEXT")
        for hit in support_hits:
            lines.append(
                f"- [{hit.get('chunk_id')}] "
                f"{_compact(str(hit.get('purpose_text') or hit.get('context_text') or ''), 500)}"
            )
    return "\n".join(lines)


def retrieve_lammps_purpose_and_command_hints(
    query: str,
    *,
    llm: Optional[Any] = None,
    top_k: int = 10,
    max_chars_per_hit: int = 1400,
) -> Dict[str, Any]:
    selector_mode = os.getenv("SIMMOF_LAMMPS_RAG_SELECTOR", "1").strip().lower()
    if selector_mode in {"0", "false", "no", "off"}:
        purpose_info = retrieve_lammps_purpose_hints(
            query,
            top_k=top_k,
            max_chars_per_hit=max_chars_per_hit,
        )
    else:
        purpose_info = retrieve_lammps_llm_selected_purpose_hints(
            query,
            llm=llm,
            max_selected=min(max(0, top_k), EVIDENCE_MAX_CORE_CHUNKS),
            max_support=EVIDENCE_MAX_SUPPORT_CHUNKS,
            max_chars_per_hit=max_chars_per_hit,
        )

    evidence_plan = build_selector_preserved_evidence_plan(
        query,
        purpose_info.get("hits") or [],
        selected_ids=purpose_info.get("selected_ids") or [],
        role_by_id=purpose_info.get("role_by_id") or {},
        max_core_commands=min(
            max(0, top_k),
            EVIDENCE_MAX_CORE_CHUNKS,
        ),
        max_dependency_commands=EVIDENCE_MAX_DEPENDENCY_COMMANDS,
    )
    selected_ids = set(purpose_info.get("selected_ids") or [])
    selector_commands = [
        str(hit.get("anchor_command") or "").lower().strip()
        for hit in purpose_info.get("hits") or []
        if str(hit.get("chunk_id") or "") in selected_ids
        and str(hit.get("anchor_command") or "").strip()
    ]
    exact_commands = list(
        dict.fromkeys(
            [
                *selector_commands,
                *(evidence_plan.get("command_names") or []),
            ]
        )
    )[:COMMAND_EVIDENCE_MAX_HITS]
    command_info = retrieve_lammps_command_hints(
        query,
        required_commands=exact_commands,
        top_k=0,
        max_chars_per_hit=min(max_chars_per_hit, 1000),
        exclude_common_boilerplate=False,
        max_common_support_hits=0,
    )
    formatted_parts = [
        _format_command_evidence_plan(evidence_plan, purpose_info),
        command_info.get("formatted_hints") or "",
    ]
    from .dependency_graph import build_advisory_dependency_graph

    dependency_graph = build_advisory_dependency_graph(evidence_plan)

    out = dict(purpose_info)
    out.update(
        {
            "exact_command_names": exact_commands,
            "evidence_plan": evidence_plan,
            "evidence_candidates": dependency_graph.get(
                "candidate_commands"
            )
            or [],
            "dependency_graph": dependency_graph,
            "protocol_changes_allowed": bool(
                evidence_plan.get("protocol_changes_allowed")
            ),
            "command_hits": command_info.get("hits", []),
            "command_support_hits": command_info.get("support_hits", []),
            "command_info": command_info,
            "formatted_hints": "\n\n".join(part for part in formatted_parts if part).strip(),
        }
    )
    return out


__all__ = [
    "build_lammps_command_evidence_plan",
    "build_selector_preserved_evidence_plan",
    "build_lammps_purpose_selector_messages",
    "load_lammps_command_summaries",
    "load_lammps_purpose_chunks",
    "retrieve_lammps_command_hints",
    "retrieve_lammps_purpose_hints",
    "retrieve_lammps_llm_selected_purpose_hints",
    "retrieve_lammps_purpose_and_command_hints",
    "select_lammps_purpose_evidence",
]
