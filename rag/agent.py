import os
import re
import json
import pickle
import faiss

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain.schema import SystemMessage, HumanMessage

from config import AGENT_LLM_MAP, LLM_DEFAULT, RAG_CORPUS_DIR, RAG_EMBED_MODEL_NAME, RAG_STORE_DIR


@dataclass
class SearchQuery:
    intent: str
    query: str
    top_n: int = 60


@dataclass
class Hit:
    filename: str
    chunk_id: int
    score: float
    text: str
    source_query: str


DEFAULT_SYSTEM = (
    "You are a helpful assistant.\n"
    "Return STRICT JSON only (no markdown, no commentary).\n"
)

TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "system_in": {
        "label": "LAMMPS system.in RUN SECTION",
        "relevant_if": [
            "ensemble choice",
            "thermostat or barostat settings",
            "timestep",
            "run length",
            "MSD or diffusivity workflow",
            "dump, thermo, or averaging settings",
            "LAMMPS run commands or protocol details",
        ],
        "not_relevant_if": [
            "experimental characterization only",
            "adsorption isotherms without MD protocol",
            "chemistry discussion without simulation protocol",
            "unrelated methods",
        ],
        "focus_notes": [
            "MD protocol",
            "LAMMPS run-section decisions",
            "diffusivity or MSD workflow if present",
        ],
        "forbidden_terms": ["pair_style", "kspace_style"],
        "min_notes_chars": 20,
    },

    "raspa_models": {
        "label": "RASPA adsorption simulation model selection",
        "relevant_if": [
            "framework forcefield selection",
            "guest molecule model or MoleculeDefinition family",
            "how guest models are represented in adsorption simulations",
            "which forcefield or model families were used in similar MOF adsorption studies",
            "actionable model-selection conventions for adsorption simulations",
        ],
        "not_relevant_if": [
            "adsorption results without model or setup details",
            "experimental characterization only",
            "general chemistry discussion without model choice",
            "MD or LAMMPS protocol unrelated to RASPA model selection",
            "screening workflow discussion without forcefield or model choice",
        ],
        "focus_notes": [
            "framework forcefield choice patterns",
            "guest model-family choice patterns",
            "clearly stated convention for this guest or MOF type",
        ],
        "forbidden_terms": [],
        "min_notes_chars": 20,
    },

    "vasp_incar": {
        "label": "VASP INCAR decision support",
        "relevant_if": [
            "dispersion usage in VASP",
            "spin or magnetism considerations",
            "plus-U usage",
            "symmetry settings",
            "smearing choices",
            "relaxation, static, DOS, or electronic-structure INCAR conventions",
        ],
        "not_relevant_if": [
            "results discussion without computational setup",
            "experimental characterization only",
            "non-VASP workflows only",
        ],
        "focus_notes": [
            "generic INCAR decision hints",
            "safe VASP setting patterns",
        ],
        "forbidden_terms": [],
        "min_notes_chars": 20,
    },

    "zeopp": {
        "label": "Zeo++ command and analysis selection",
        "relevant_if": [
            "accessible surface area or accessible volume analysis",
            "geometric pore volume analysis",
            "PLD or LCD style metrics",
            "channel or cavity analysis",
            "probe-radius choice logic",
        ],
        "not_relevant_if": [
            "adsorption results without geometric-analysis setup",
            "experimental characterization only",
            "general discussion without Zeo++-style analysis choice",
        ],
        "focus_notes": [
            "which Zeo++ analyses are useful",
            "probe-radius choice logic",
            "metric-selection patterns",
        ],
        "forbidden_terms": [],
        "min_notes_chars": 20,
    },

    "screening": {
        "label": "high-throughput MOF screening workflow design",
        "relevant_if": [
            "screening workflow order",
            "tool-chain selection",
            "geometric prefiltering",
            "Henry coefficient ranking",
            "expensive refinement after early screening",
            "workflow design rationale",
        ],
        "not_relevant_if": [
            "single-study results without workflow design",
            "experimental characterization only",
            "discussion unrelated to screening pipelines",
        ],
        "focus_notes": [
            "workflow step order",
            "tool-chain patterns",
            "screening rationale",
        ],
        "forbidden_terms": [],
        "min_notes_chars": 20,
    },
}


GENERIC_GATE_PROMPT = """
You will be given:
- Target simulation context (MOF, guest, property, description)
- A target task/tool
- Task-specific relevance criteria
- Snippets from ONE retrieved document (paper text chunks or file excerpt)

Task:
1) Decide if this document is ACTUALLY relevant for the target task.
2) Relevant ONLY if the snippets contain actionable information matching the provided relevance criteria.
3) If relevant, write ONLY useful notes as concise bullet lines (2-4 lines).
4) If snippets do not contain actionable information for this task, mark not relevant.

Context:
MOF = {mof}
Guest = {guest}
Property = {property}
Simulation description = {simulation_description}

Target task:
{task_label}

Relevant if the document contains actionable information about:
{relevant_if}

Not relevant if the document is mainly about:
{not_relevant_if}

If relevant, focus notes ONLY on:
{focus_notes}

Document:
filename = {filename}

Snippets:
{snippets}

Return ONLY JSON:
{{
  "is_relevant": true/false,
  "notes": ".... (empty string if not relevant)"
}}
""".strip()


class RagAgent:
    def __init__(
        self,
        store_dir: str = str(RAG_STORE_DIR),
        embed_model_name: str = RAG_EMBED_MODEL_NAME,
        llm=None,
        agent_name: str = "RagAgent",
        *,
        per_query_topn: int = 60,
        final_papers: int = 5,
        max_chars_per_evidence: int = 900,
        corpus_dir: str = str(RAG_CORPUS_DIR),
    ):
        self.agent_name = agent_name
        self.llm = llm if llm is not None else AGENT_LLM_MAP.get(agent_name, LLM_DEFAULT)

        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing index.faiss: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing metadata.pkl: {meta_path}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta: List[Dict[str, Any]] = pickle.load(f)

        if self.index.ntotal != len(self.meta):
            raise RuntimeError(
                f"Index/metadata mismatch: index.ntotal={self.index.ntotal} vs meta={len(self.meta)}"
            )

        self.embedder = SentenceTransformer(embed_model_name)
        self.metric = self._detect_metric()

        self.per_query_topn = per_query_topn
        self.final_papers = final_papers
        self.max_chars_per_evidence = max_chars_per_evidence

        self.corpus_dir = corpus_dir

    def _call_llm(self, messages: List[Any]) -> str:
        llm_obj = self.llm

        if callable(llm_obj) and not hasattr(llm_obj, "invoke"):
            try:
                candidate = llm_obj()
                if hasattr(candidate, "invoke"):
                    llm_obj = candidate
            except TypeError:
                pass

        if hasattr(llm_obj, "invoke"):
            resp = llm_obj.invoke(messages)
            return str(getattr(resp, "content", resp))
        else:
            resp = llm_obj(messages)
            if hasattr(resp, "content"):
                return str(resp.content)
            return str(resp)

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        t = str(text).strip()
        if t.startswith("```"):
            lines = t.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()

        try:
            return json.loads(t)
        except Exception:
            l = t.find("{")
            r = t.rfind("}")
            if 0 <= l < r:
                return json.loads(t[l : r + 1])
            raise

    def run(
        self,
        user_text: str,
        parsed_query: Optional[Dict[str, Any]] = None,
        *,
        k_papers: Optional[int] = None,
    ) -> Dict[str, Any]:
        parsed_query = parsed_query or {}
        k_papers = k_papers or self.final_papers

        queries = self.plan_queries_with_llm(user_text, parsed_query)
        if not queries:
            queries = [SearchQuery(intent="main", query=user_text, top_n=self.per_query_topn)]

        pool = self.retrieve_pool(queries)
        pool = self.dedup_by_chunk(pool)
        top_papers = self.select_diverse_papers(pool, k=k_papers)
        evidence_block = self.format_evidence_block(top_papers)

        return {
            "metric": self.metric,
            "queries": [q.__dict__ for q in queries],
            "top_papers": top_papers,
            "evidence_block": evidence_block,
        }

    def _dedup_queries(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        seen = set()
        out: List[SearchQuery] = []
        for q in queries:
            key = (q.query or "").lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out


    def _make_simulation_description(
        self,
        ctx: Dict[str, Any],
        default_text: str,
    ) -> str:
        query_text = str(ctx.get("query_text", "")).strip()
        job_name = str(ctx.get("job_name", "")).strip()

        if query_text:
            return f"[JOB_NAME={job_name}] {query_text}" if job_name else query_text
        return default_text


    def _build_file_notes_for_task(
        self,
        *,
        task_name: str,
        ctx: Dict[str, Any],
        simulation_description: str,
        top_file_hits: List[Dict[str, Any]],
        per_file_max_snippets: int = 6,
        per_snippet_max_chars: int = 900,
        file_read_max_chars: int = 12000,
        file_excerpt_chars: int = 2000,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        mof = str(ctx.get("mof", "")).strip()
        guest = str(ctx.get("guest", "")).strip()
        prop = str(ctx.get("property", "")).strip()

        file_notes: List[Dict[str, Any]] = []
        relevant_blocks: List[str] = []

        for item in top_file_hits:
            filename = item["filename"]
            hits: List[Hit] = item["hits"]

            snippets = self._build_snippets_for_file(
                filename=filename,
                hits=hits,
                max_snippets=per_file_max_snippets,
                max_chars=per_snippet_max_chars,
                file_read_max_chars=file_read_max_chars,
                file_excerpt_chars=file_excerpt_chars,
            )

            if not snippets.strip():
                note_obj = {"filename": filename, "is_relevant": False, "notes": ""}
                file_notes.append(note_obj)
                continue

            note_obj = self._llm_gate_by_task(
                task_name=task_name,
                mof=mof,
                guest=guest,
                prop=prop,
                simulation_description=simulation_description,
                filename=filename,
                snippets=snippets,
            )
            file_notes.append(note_obj)

            notes = str(note_obj.get("notes") or "").strip()
            if note_obj.get("is_relevant") and notes:
                relevant_blocks.append(f"file={filename}\n{notes}")

        return file_notes, relevant_blocks


    def _run_task_pipeline(
        self,
        *,
        task_name: str,
        ctx: Dict[str, Any],
        queries: List[SearchQuery],
        simulation_description: str,
        top_files: int = 5,
        per_file_max_snippets: int = 6,
        per_snippet_max_chars: int = 900,
        file_read_max_chars: int = 12000,
        file_excerpt_chars: int = 2000,
    ) -> Dict[str, Any]:
        queries = self._dedup_queries(queries)

        pool = self.retrieve_pool(queries)
        pool = self.dedup_by_chunk(pool)
        top_file_hits = self._select_top_files_with_hits(pool, k=top_files)

        file_notes, relevant_blocks = self._build_file_notes_for_task(
            task_name=task_name,
            ctx=ctx,
            simulation_description=simulation_description,
            top_file_hits=top_file_hits,
            per_file_max_snippets=per_file_max_snippets,
            per_snippet_max_chars=per_snippet_max_chars,
            file_read_max_chars=file_read_max_chars,
            file_excerpt_chars=file_excerpt_chars,
        )

        evidence_block = "\n\n".join(relevant_blocks).strip()

        return {
        "queries": [q.__dict__ for q in queries],
        "pool": pool,
        "top_file_hits": top_file_hits,
        "top_papers": self.select_diverse_papers(pool, k=top_files),
        "file_notes": file_notes,
        "evidence_block": evidence_block,
    }

    def run_for_system_in(
        self,
        context: Dict[str, Any],
        *,
        top_files: int = 5,
        per_file_max_snippets: int = 6,
        per_snippet_max_chars: int = 900,
        file_read_max_chars: int = 12000,
        file_excerpt_chars: int = 2000,
    ) -> Dict[str, Any]:
        mof = str(context.get("mof", "")).strip()
        guest = str(context.get("guest", "")).strip()
        prop = str(context.get("property", "")).strip()

        simulation_description = self._make_simulation_description(
            context,
            f"Compute {prop} of {guest} in {mof}",
        )

        queries = self._build_systemin_queries(mof, guest, prop, simulation_description)

        base = self._run_task_pipeline(
            task_name="system_in",
            ctx=context,
            queries=queries,
            simulation_description=simulation_description,
            top_files=top_files,
            per_file_max_snippets=per_file_max_snippets,
            per_snippet_max_chars=per_snippet_max_chars,
            file_read_max_chars=file_read_max_chars,
            file_excerpt_chars=file_excerpt_chars,
        )

        useful_blocks: List[str] = []
        for x in base["file_notes"]:
            notes = str(x.get("notes") or "").strip()
            if x.get("is_relevant") and notes:
                useful_blocks.append(f'file={x["filename"]}\n{notes}')

        rag_summaries = "\n\n".join(useful_blocks).strip()

        return {
            "systemin_queries": base["queries"],
            "file_notes": base["file_notes"],
            "rag_summaries": rag_summaries,
        }

    def _build_systemin_queries(self, mof: str, guest: str, prop: str, sim_desc: str) -> List[SearchQuery]:
        base = f"{mof} {guest} {prop}".strip()
        return [
            SearchQuery(intent="main", query=sim_desc, top_n=self.per_query_topn),
            SearchQuery(intent="main", query=f"{base} molecular dynamics LAMMPS input run section", top_n=self.per_query_topn),
            SearchQuery(intent="explain", query=f"{base} diffusion compute msd LAMMPS fix langevin nve dump thermo", top_n=self.per_query_topn),
            SearchQuery(intent="optional", query=f"{mof} {guest} diffusion coefficient simulation timestep run length LAMMPS", top_n=self.per_query_topn),
        ]

    def _select_top_files_with_hits(self, hits: List[Hit], k: int = 5) -> List[Dict[str, Any]]:
        by_file: Dict[str, List[Hit]] = {}
        for h in hits:
            if not h.filename:
                continue
            by_file.setdefault(h.filename, []).append(h)

        def best_score(hs: List[Hit]) -> float:
            return max(x.score for x in hs) if self.metric == "ip" else min(x.score for x in hs)

        files: List[Dict[str, Any]] = []
        for fn, hs in by_file.items():
            hs_sorted = sorted(hs, key=lambda x: x.score, reverse=(self.metric == "ip"))
            files.append({"filename": fn, "best": best_score(hs), "hits": hs_sorted})

        files = sorted(files, key=lambda x: x["best"], reverse=(self.metric == "ip"))
        return files[:k]

    def _resolve_txt_path(self, filename: str) -> Optional[str]:
        if not filename:
            return None

        p = os.path.join(self.corpus_dir, filename)
        if os.path.exists(p):
            return p

        base = os.path.basename(filename)
        p = os.path.join(self.corpus_dir, base)
        if os.path.exists(p):
            return p

        if not base.endswith(".txt"):
            p = os.path.join(self.corpus_dir, base + ".txt")
            if os.path.exists(p):
                return p

        return None

    def _read_text_file(self, path: str, max_chars: int = 12000) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read(max_chars)
            return (txt or "").strip()
        except Exception:
            return ""

    def _build_snippets_for_file(
        self,
        *,
        filename: str,
        hits: List[Hit],
        max_snippets: int = 6,
        max_chars: int = 900,
        file_read_max_chars: int = 12000,
        file_excerpt_chars: int = 2000,
    ) -> str:
        lines: List[str] = []

        txt_path = self._resolve_txt_path(filename)
        if txt_path:
            head = self._read_text_file(txt_path, max_chars=file_read_max_chars)
            if head:
                head = re.sub(r"\s+", " ", head).strip()
                excerpt = head[:file_excerpt_chars]
                lines.append(f"[file_excerpt path={txt_path}] {excerpt}")
                lines.append("")

        for h in hits[:max_snippets]:
            txt = self._compact_text(h.text, max_chars=max_chars)
            if txt:
                lines.append(f"[chunk_id={h.chunk_id} score={h.score:.4f}] {txt}")

        return "\n".join(lines).strip()

    def _llm_gate_by_task(
        self,
        *,
        task_name: str,
        mof: str,
        guest: str,
        prop: str,
        simulation_description: str,
        filename: str,
        snippets: str,
    ) -> Dict[str, Any]:
        spec = TASK_SPECS.get(task_name)
        if not spec:
            return {"filename": filename, "is_relevant": False, "notes": ""}

        prompt = GENERIC_GATE_PROMPT.format(
            mof=mof,
            guest=guest,
            property=prop,
            simulation_description=simulation_description,
            task_label=spec["label"],
            relevant_if="\n".join(f"- {x}" for x in spec["relevant_if"]),
            not_relevant_if="\n".join(f"- {x}" for x in spec["not_relevant_if"]),
            focus_notes="\n".join(f"- {x}" for x in spec["focus_notes"]),
            filename=filename,
            snippets=snippets,
        )

        messages = [
            SystemMessage(content="Return STRICT JSON only (no markdown, no commentary)."),
            HumanMessage(content=prompt),
        ]
        raw = self._call_llm(messages)

        try:
            obj = self._safe_json_loads(raw)
        except Exception:
            return {"filename": filename, "is_relevant": False, "notes": ""}

        is_rel = bool(obj.get("is_relevant", False))
        notes = str(obj.get("notes") or "").strip()

        min_notes_chars = int(spec.get("min_notes_chars", 20))
        if is_rel and len(notes) < min_notes_chars:
            return {"filename": filename, "is_relevant": False, "notes": ""}

        for banned in spec.get("forbidden_terms", []):
            if banned.lower() in notes.lower():
                return {"filename": filename, "is_relevant": False, "notes": ""}

        return {
            "filename": filename,
            "is_relevant": is_rel,
            "notes": notes,
        }

    def run_for_raspa_models(
        self,
        ctx: Dict[str, Any],
        top_files: int = 5,
        per_file_max_snippets: int = 6,
        per_snippet_max_chars: int = 900,
        file_read_max_chars: int = 12000,
        file_excerpt_chars: int = 2000,
    ) -> dict:
        mof = (ctx.get("mof") or "").strip()
        guest = (ctx.get("guest") or "").strip()
        prop = (ctx.get("property") or "").strip()

        simulation_description = self._make_simulation_description(
            ctx,
            f"RASPA adsorption simulation for {guest} in {mof} ({prop})",
        )

        queries = [
            SearchQuery(
                intent="main",
                query=(
                    f"RASPA adsorption simulation force field selection and guest model definition "
                    f"for MOF {mof} with guest {guest}. Property/task: {prop}. "
                    f"{str(ctx.get('query_text', '')).strip()}"
                ).strip(),
                top_n=self.per_query_topn,
            ),
            SearchQuery(
                intent="main",
                query=f"{mof} {guest} adsorption force field model RASPA",
                top_n=self.per_query_topn,
            ),
            SearchQuery(
                intent="explain",
                query=f"{mof} {guest} guest model adsorption simulation molecule definition",
                top_n=self.per_query_topn,
            ),
            SearchQuery(
                intent="optional",
                query=f"{mof} framework forcefield adsorption simulation guest model",
                top_n=self.per_query_topn,
            ),
        ]

        base = self._run_task_pipeline(
            task_name="raspa_models",
            ctx=ctx,
            queries=queries,
            simulation_description=simulation_description,
            top_files=top_files,
            per_file_max_snippets=per_file_max_snippets,
            per_snippet_max_chars=per_snippet_max_chars,
            file_read_max_chars=file_read_max_chars,
            file_excerpt_chars=file_excerpt_chars,
        )

        relevant_docs: List[Dict[str, Any]] = []
        top_file_hits = base["top_file_hits"]

        notes_by_filename = {
            str(x.get("filename")): x
            for x in base["file_notes"]
        }

        for item in top_file_hits:
            filename = item["filename"]
            note_obj = notes_by_filename.get(filename, {})
            if not note_obj.get("is_relevant"):
                continue

            hits: List[Hit] = item["hits"]
            snippets = self._build_snippets_for_file(
                filename=filename,
                hits=hits,
                max_snippets=per_file_max_snippets,
                max_chars=per_snippet_max_chars,
                file_read_max_chars=file_read_max_chars,
                file_excerpt_chars=file_excerpt_chars,
            )

            summary_text_parts: List[str] = []

            notes = str(note_obj.get("notes") or "").strip()
            if notes:
                summary_text_parts.append(f"[RAG relevance notes]\n{notes}")

            if snippets:
                summary_text_parts.append(f"[Retrieved snippets]\n{snippets}")

            txt_path = self._resolve_txt_path(filename)
            if txt_path:
                txt = self._read_text_file(txt_path, max_chars=12000)
                txt = " ".join((txt or "").split()).strip()
                if txt:
                    summary_text_parts.append(f"[File excerpt]\n{txt[:4000]}")

            merged_text = "\n\n".join(summary_text_parts).strip()
            if merged_text:
                relevant_docs.append({
                    "filename": filename,
                    "text": merged_text[:22000],
                })

        if not relevant_docs:
            return {
                "raspa_model_queries": base["queries"],
                "file_notes": base["file_notes"],
                "forcefield_hints": "",
                "molecule_hints": "",
            }

        forcefield_hints = self._summarize_raspa_hints(
            relevant_docs,
            focus="forcefield",
            ctx=ctx,
        )

        molecule_hints = self._summarize_raspa_hints(
            relevant_docs,
            focus="molecule_definition",
            ctx=ctx,
        )

        return {
            "raspa_model_queries": base["queries"],
            "file_notes": base["file_notes"],
            "forcefield_hints": forcefield_hints.strip(),
            "molecule_hints": molecule_hints.strip(),
        }

    def run_for_screening_workflows(
        self,
        ctx: Dict[str, Any],
        top_files: int = 5,
        per_file_max_snippets: int = 6,
        per_snippet_max_chars: int = 900,
        file_read_max_chars: int = 12000,
        file_excerpt_chars: int = 2000,
    ) -> dict:
        mof = (ctx.get("mof") or "").strip()
        guest = (ctx.get("guest") or "").strip()
        prop = (ctx.get("property") or "").strip()

        simulation_description = self._make_simulation_description(
            ctx,
            f"high-throughput screening workflow for {guest} in {mof} targeting {prop}",
        )

        queries = [
            SearchQuery(intent="main", query=simulation_description, top_n=self.per_query_topn),
            SearchQuery(
                intent="main",
                query=f"high-throughput screening workflow for MOF {mof} with guest {guest} target property {prop}".strip(),
                top_n=self.per_query_topn,
            ),
            SearchQuery(
                intent="explain",
                query="MOF screening workflow geometric filter Henry coefficient GCMC ranking",
                top_n=self.per_query_topn,
            ),
            SearchQuery(
                intent="optional",
                query="MOF screening tool chain Zeo++ Henry adsorption refinement",
                top_n=self.per_query_topn,
            ),
        ]

        base = self._run_task_pipeline(
            task_name="screening",
            ctx=ctx,
            queries=queries,
            simulation_description=simulation_description,
            top_files=top_files,
            per_file_max_snippets=per_file_max_snippets,
            per_snippet_max_chars=per_snippet_max_chars,
            file_read_max_chars=file_read_max_chars,
            file_excerpt_chars=file_excerpt_chars,
        )

        workflow_hints = self._summarize_screening_workflow_hints_with_llm(
            ctx=ctx,
            evidence_block=base["evidence_block"],
        )

        return {
            "top_papers": base["top_papers"],
            "file_notes": base["file_notes"],
            "workflow_hints": workflow_hints.strip(),
        }

    def _summarize_raspa_hints(self, docs: list, focus: str, ctx: Dict[str, Any]) -> str:
        focus_text = "framework forcefield choice" if focus == "forcefield" else "guest molecule definition/model choice"

        system_msg = (
            "You are summarizing ONLY actionable hints for RASPA input model selection.\n"
            "Output MUST be 2–4 bullet lines starting with '- '.\n"
            "Bullets must be generic and non-numeric.\n"
            "Avoid long run lengths, avoid cycles/probabilities, avoid parameter tuning.\n"
            "Only mention model/choice patterns (framework forcefield vs guest model families) if clearly supported.\n"
            "If nothing clearly relevant is found, output an empty string.\n"
            "No extra text."
        )

        mof = ctx.get("mof", "")
        guest = ctx.get("guest", "")
        prop = ctx.get("property", "")

        joined = []
        for d in docs:
            joined.append(f"FILE={d['filename']}\nTEXT={d['text']}\n")
        corpus = "\n---\n".join(joined)

        prompt = (
            f"Task: extract RASPA hints about {focus_text}.\n"
            f"Context: MOF={mof}, guest={guest}, property={prop}.\n"
            f"Return only the useful notes for {focus_text} as concise bullets (2-4 lines).\n\n"
            f"Evidence:\n{corpus}"
        )

        messages = [SystemMessage(content=system_msg), HumanMessage(content=prompt)]
        raw = self._call_llm(messages)
        text = str(raw).strip()

        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()

        if not text or not any(ln.strip().startswith("- ") for ln in text.splitlines()):
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
        return "\n".join(lines[:4])

    def _summarize_screening_workflow_hints_with_llm(self, ctx: Dict[str, Any], evidence_block: str) -> str:
        mof = (ctx.get("mof") or "").strip()
        guest = (ctx.get("guest") or "").strip()
        prop = (ctx.get("property") or "").strip()
        qtxt = (ctx.get("query_text") or "").strip()

        system_msg = (
            "You extract ONLY useful, generic hints for designing a minimal screening workflow for MOFs.\n"
            "Output MUST be 2–6 bullet lines starting with '- '.\n"
            "Hard constraints:\n"
            "- If evidence is not relevant to screening workflows, output an empty string.\n"
            "- Avoid numeric parameters (no exact temperatures, pressures, cutoffs, cycle numbers).\n"
            "- Focus on WHICH steps/tools are used and WHY, as a tool-chain (e.g., geometric prefilter -> Henry ranking -> expensive simulation).\n"
            "- Do NOT invent tools outside common MOF screening context.\n"
            "- Do NOT paste long sentences from evidence.\n"
            "No extra text."
        )

        prompt = f"""
Task: extract screening workflow design hints (tool-chain + rationale).

Context:
MOF={mof}
Guest={guest}
Property={prop}
User query={qtxt}

What to extract (examples):
- Typical order of screening steps (geometry prefilter -> cheap adsorption proxy -> expensive refinement).
- When Henry coefficient ranking is used vs full GCMC/isotherm.
- When geometric filters (PLD/LCD/ASA/AV) are used as early pruning.
- Any caution about over-filtering / keeping a larger candidate pool before expensive simulations (conceptual, no numbers).

RAG evidence:
{evidence_block}
""".strip()

        raw = self._call_llm([
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt),
        ]).strip()

        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1]).strip()

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        bullets = [ln for ln in lines if ln.startswith("- ")]
        return "\n".join(bullets[:6]).strip()

    def run_for_vasp_incar(
        self,
        rag_ctx: Dict[str, Any],
        top_files: int = 5,
        per_file_max_snippets: int = 6,
        per_snippet_max_chars: int = 900,
        file_read_max_chars: int = 12000,
        file_excerpt_chars: int = 2000,
    ) -> Dict[str, Any]:
        mof = (rag_ctx.get("mof") or "").strip()
        guest = (rag_ctx.get("guest") or "").strip()
        prop = (rag_ctx.get("property") or "").strip()

        simulation_description = self._make_simulation_description(
            rag_ctx,
            f"VASP INCAR settings for {prop} of {guest} in {mof}",
        )

        queries = [
            SearchQuery(intent="main", query=simulation_description, top_n=self.per_query_topn),
            SearchQuery(intent="main", query=f"{mof} {guest} VASP INCAR settings {prop}", top_n=self.per_query_topn),
            SearchQuery(intent="explain", query=f"{mof} VASP dispersion spin U INCAR settings", top_n=self.per_query_topn),
            SearchQuery(intent="optional", query=f"{mof} VASP relax static DOS symmetry smearing", top_n=self.per_query_topn),
        ]

        base = self._run_task_pipeline(
            task_name="vasp_incar",
            ctx=rag_ctx,
            queries=queries,
            simulation_description=simulation_description,
            top_files=top_files,
            per_file_max_snippets=per_file_max_snippets,
            per_snippet_max_chars=per_snippet_max_chars,
            file_read_max_chars=file_read_max_chars,
            file_excerpt_chars=file_excerpt_chars,
        )

        vasp_incar_hints = self._summarize_vasp_incar_hints_with_llm(
            rag_ctx=rag_ctx,
            evidence_block=base["evidence_block"],
        )

        return {
            "top_papers": base["top_papers"],
            "file_notes": base["file_notes"],
            "vasp_incar_hints": vasp_incar_hints,
        }

    def _summarize_vasp_incar_hints_with_llm(self, rag_ctx: Dict[str, Any], evidence_block: str) -> str:
        mof = (rag_ctx.get("mof") or "").strip()
        guest = (rag_ctx.get("guest") or "").strip()
        prop = (rag_ctx.get("property") or "").strip()
        stage = (rag_ctx.get("vasp_stage") or "").strip()
        calc = (rag_ctx.get("vasp_calc_type") or "").strip()
        qtxt = (rag_ctx.get("query_text") or "").strip()

        prompt = f"""
You are extracting ONLY useful, generic VASP INCAR hints for a MOF simulation.

Task:
Given RAG evidence (paper chunks), write ONLY 2–4 bullet lines that could help choose safe INCAR tags.

Hard constraints:
- Output ONLY bullets (each line starts with "- ").
- If evidence is not relevant to INCAR choices, output an empty string.
- Avoid numbers (ENCUT/EDIFF/etc). No run lengths. No k-point grids.
- Focus ONLY on generic INCAR decision hints, like:
  * dispersion usage (include IVDW or not)
  * spin / magnetism considerations (ISPIN/MAGMOM) when metals/open-shell suspected
  * +U possibility (LDAU) only if clearly indicated
  * symmetry / smearing practices for MOFs (ISYM, ISMEAR/SIGMA) in relax vs DOS
  * DOS/static cues (IBRION/NSW/ICHARG/LORBIT) if stage implies it
- Do NOT copy long sentences from evidence.

Context:
MOF={mof}
Guest={guest}
Property={prop}
VASP_STAGE={stage}
VASP_CALC_TYPE={calc}
User query={qtxt}

RAG evidence:
{evidence_block}
""".strip()

        messages = [
            SystemMessage(content="You are a careful scientific assistant. Output only the requested bullets."),
            HumanMessage(content=prompt),
        ]
        raw = self._call_llm(messages).strip()

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        bullets = [ln for ln in lines if ln.startswith("- ")]
        return "\n".join(bullets[:4]).strip()


    def run_for_zeopp(
        self,
        rag_ctx: Dict[str, Any],
        top_files: int = 5,
        per_file_max_snippets: int = 6,
        per_snippet_max_chars: int = 900,
        file_read_max_chars: int = 12000,
        file_excerpt_chars: int = 2000,
    ) -> Dict[str, Any]:
        mof = (rag_ctx.get("mof") or "").strip()
        guest = (rag_ctx.get("guest") or "").strip()
        prop = (rag_ctx.get("property") or "").strip()

        simulation_description = self._make_simulation_description(
            rag_ctx,
            f"Zeo++ analysis for {prop} of {guest} in {mof}",
        )

        queries = [
            SearchQuery(intent="main", query=simulation_description, top_n=self.per_query_topn),
            SearchQuery(intent="main", query=f"{mof} {guest} Zeo++ pore analysis {prop}", top_n=self.per_query_topn),
            SearchQuery(intent="explain", query=f"{mof} Zeo++ PLD LCD surface area accessible volume probe radius", top_n=self.per_query_topn),
            SearchQuery(intent="optional", query=f"{mof} channel cavity analysis pore limiting diameter largest cavity diameter", top_n=self.per_query_topn),
        ]

        base = self._run_task_pipeline(
            task_name="zeopp",
            ctx=rag_ctx,
            queries=queries,
            simulation_description=simulation_description,
            top_files=top_files,
            per_file_max_snippets=per_file_max_snippets,
            per_snippet_max_chars=per_snippet_max_chars,
            file_read_max_chars=file_read_max_chars,
            file_excerpt_chars=file_excerpt_chars,
        )

        zeopp_hints = self._summarize_zeopp_hints_with_llm(
            rag_ctx,
            base["evidence_block"],
        )

        return {
            "top_papers": base["top_papers"],
            "file_notes": base["file_notes"],
            "zeopp_hints": zeopp_hints,
        }

    def _summarize_zeopp_hints_with_llm(self, rag_ctx: Dict[str, Any], evidence_block: str) -> str:
        mof = (rag_ctx.get("mof") or "").strip()
        guest = (rag_ctx.get("guest") or "").strip()
        prop = (rag_ctx.get("property") or "").strip()
        qtxt = (rag_ctx.get("query_text") or "").strip()

        prompt = f"""
You are extracting ONLY useful, generic Zeo++ (zeopp) command selection hints for MOF geometric analysis.

Task:
Given RAG evidence (paper chunks), write ONLY 2–4 bullet lines that help choose Zeo++ command/flags.

Hard constraints:
- Output ONLY bullets (each line starts with "- ").
- If evidence is not relevant to Zeo++ command choices, output an empty string.
- Avoid long numeric settings. If a numeric is mentioned, keep it generic (e.g., "use a probe radius appropriate to the guest size").
- Focus ONLY on which *type* of Zeo++ analysis to run, e.g.:
  * accessible surface area / accessible volume vs geometric pore volume
  * channel/accessible network analysis vs cavity analysis
  * PLD/LCD style metrics (pore limiting / largest cavity) if supported by evidence
  * probe-radius choice logic (guest-sized vs generic N2/He probes) as a concept

Context:
MOF={mof}
Guest={guest}
Property={prop}
User query={qtxt}

RAG evidence:
{evidence_block}
""".strip()

        raw = self._call_llm([
            SystemMessage(content="Output only the requested bullets. No extra text."),
            HumanMessage(content=prompt),
        ]).strip()

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        bullets = [ln for ln in lines if ln.startswith("- ")]
        return "\n".join(bullets[:4]).strip()

    def plan_queries_with_llm(self, user_text: str, parsed_query: Dict[str, Any]) -> List[SearchQuery]:
        prompt = self._build_querygen_prompt(user_text, parsed_query)

        messages = [
            SystemMessage(content=DEFAULT_SYSTEM),
            HumanMessage(content=prompt),
        ]
        raw = self._call_llm(messages)

        try:
            obj = self._safe_json_loads(raw)
        except Exception:
            return []

        if not isinstance(obj, dict):
            return []
        items = obj.get("queries")
        if not isinstance(items, list):
            return []

        out: List[SearchQuery] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            q = str(it.get("query") or "").strip()
            if not q:
                continue
            intent = str(it.get("intent") or "main").strip()
            top_n = it.get("top_n")
            if not isinstance(top_n, int) or top_n <= 0:
                top_n = self.per_query_topn
            out.append(SearchQuery(intent=intent, query=q, top_n=top_n))

        return self._dedup_queries(out)

    def _build_querygen_prompt(self, user_text: str, parsed_query: Dict[str, Any]) -> str:
        hint = json.dumps(parsed_query, ensure_ascii=False)

        return f"""
You are a "RAG search query generator" for computational chemistry workflows.

Goal:
Generate 3–6 search queries that will retrieve paper excerpts describing
WHAT computational analyses/simulations are used to solve the user's task.

Constraints:
- Do NOT include specific numeric parameters or settings.
- Do NOT hardcode a named technique (e.g., do not force a specific charge partition method).
- Use general, tool-level / method-family phrases when helpful (examples: first-principles/DFT, adsorption/binding/interaction energy,
  electronic structure analysis, charge analysis, adsorption simulation (Monte Carlo), pore characterization, molecular dynamics).
- At least ONE query must be the original user question verbatim.
- Keep queries short and keyword-rich.

Output ONLY valid JSON (no markdown).
Schema:
{{
  "queries": [
    {{"intent":"main|explain|optional", "query":"...", "top_n": {self.per_query_topn}}}
  ]
}}

User question:
{user_text}

Structured hint (may be empty):
{hint}
""".strip()

    def retrieve_pool(self, queries: List[SearchQuery]) -> List[Hit]:
        pool: List[Hit] = []
        for q in queries:
            results = self.search(q.query, top_k=q.top_n)
            for score, idx in results:
                m = self.meta[idx]
                pool.append(
                    Hit(
                        filename=str(m.get("filename", "")),
                        chunk_id=int(m.get("chunk_id", -1)),
                        score=float(score),
                        text=str(m.get("text", "")),
                        source_query=q.query,
                    )
                )
        return pool

    def search(self, query: str, top_k: int = 60) -> List[Tuple[float, int]]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        out: List[Tuple[float, int]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            out.append((float(score), int(idx)))
        return out


    def dedup_by_chunk(self, hits: List[Hit]) -> List[Hit]:
        best: Dict[Tuple[str, int], Hit] = {}
        for h in hits:
            key = (h.filename, h.chunk_id)
            if key not in best:
                best[key] = h
                continue

            if self.metric == "ip":
                if h.score > best[key].score:
                    best[key] = h
            else:
                if h.score < best[key].score:
                    best[key] = h

        if self.metric == "ip":
            return sorted(best.values(), key=lambda x: x.score, reverse=True)
        return sorted(best.values(), key=lambda x: x.score)

    def select_diverse_papers(self, hits: List[Hit], k: int = 5) -> List[Dict[str, Any]]:
        best_by_file: Dict[str, Hit] = {}
        for h in hits:
            if not h.filename:
                continue
            if h.filename not in best_by_file:
                best_by_file[h.filename] = h
                continue

            if self.metric == "ip":
                if h.score > best_by_file[h.filename].score:
                    best_by_file[h.filename] = h
            else:
                if h.score < best_by_file[h.filename].score:
                    best_by_file[h.filename] = h

        anchors = list(best_by_file.values())
        anchors = sorted(anchors, key=lambda x: x.score, reverse=(self.metric == "ip"))
        anchors = anchors[:k]

        out: List[Dict[str, Any]] = []
        for eid, h in enumerate(anchors):
            out.append(
                {
                    "evidence_id": eid,
                    "filename": h.filename,
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "text": self._compact_text(h.text, self.max_chars_per_evidence),
                    "source_query": h.source_query,
                }
            )
        return out

    def format_evidence_block(self, top_papers: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        lines.append("EVIDENCE (top papers/chunks from RAG):")
        for p in top_papers:
            lines.append(
                f'[{p["evidence_id"]}] filename={p["filename"]}  chunk_id={p["chunk_id"]}  score={p["score"]:.4f}'
            )
            lines.append(p["text"])
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _detect_metric(self) -> str:
        t = str(type(self.index)).lower()
        if "indexflatip" in t:
            return "ip"
        if "indexflatl2" in t:
            return "l2"
        if hasattr(self.index, "metric_type"):
            mt = int(self.index.metric_type)
            if mt == faiss.METRIC_INNER_PRODUCT:
                return "ip"
            if mt == faiss.METRIC_L2:
                return "l2"
        return "ip"

    @staticmethod
    def _compact_text(text: str, max_chars: int) -> str:
        t = re.sub(r"\s+", " ", (text or "").strip())
        if len(t) <= max_chars:
            return t
        return t[: max_chars - 3] + "..."


if __name__ == "__main__":
    agent = RagAgent(agent_name="RagAgent")

    user_q = "Compute the CO2 binding energies for HKUST-1 and ZIF-8 and discuss why the two MOFs show different binding strengths"
    out = agent.run(user_q, parsed_query={"MOF": "HKUST-1, ZIF-8", "Guest": "CO2"})
    print("metric:", out["metric"])
    print("queries:", out["queries"])
    print(out["evidence_block"])

    ctx = {
        "job_name": "HKUST-1_CO2_diffusivity",
        "mof": "HKUST-1",
        "guest": "CO2",
        "property": "diffusivity",
        "query_text": "I want to calculate diffusivity of CO2 in HKUST-1",
    }
    out2 = agent.run_for_system_in(ctx, top_files=5)
    print("\n--- system.in RAG file_notes ---")
    print(json.dumps(out2["file_notes"], indent=2, ensure_ascii=False))
    print("\n--- system.in RAG summaries ---")
    print(out2["rag_summaries"])
