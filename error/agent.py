from pathlib import Path
from typing import Iterable, Optional
import re

from langchain.schema import HumanMessage, SystemMessage


class ErrorAgent:
    def _init_error_agent(self, *, llm=None, default_llm=None, max_lines: int = 200) -> None:
        self.llm = llm or default_llm
        self.max_lines = max_lines

    def read_file(self, filepath: str) -> str:
        try:
            with open(filepath, "r", errors="ignore") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return f"<< {filepath} not found >>\n"

        if len(lines) > self.max_lines:
            half = self.max_lines // 2
            lines = lines[:half] + ["\n...\n"] + lines[-half:]
        return "".join(lines)

    def _read_file(self, filepath: str) -> str:
        return self.read_file(filepath)

    def _clear_flags(self, work_dir: Path, flags: Iterable[str] = ("START", "DONE", "FAILED")) -> None:
        for fn in flags:
            p = work_dir / fn
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    def _is_finished(
        self,
        work_dir: Path,
        done_flag: str = "DONE",
        fail_flag: str = "FAILED",
    ) -> bool:
        return (work_dir / done_flag).exists() or (work_dir / fail_flag).exists()

    def _which_flag(
        self,
        work_dir: Path,
        done_flag: str = "DONE",
        fail_flag: str = "FAILED",
    ) -> Optional[str]:
        if (work_dir / done_flag).exists():
            return done_flag
        if (work_dir / fail_flag).exists():
            return fail_flag
        return None

    def _invoke_llm(self, system_prompt: str, user_prompt: str,
                    agent: str = None, label: str = None) -> str:
        if agent and label:
            from core.llm_logging import set_llm_context
            set_llm_context(agent, label)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        return (response.content or "").strip()

    def _ask_user_confirmation(self, label: str, proposed_text: str,
                                system_prompt: str = "", user_prompt: str = ""):
        from config import ask_user_confirmation

        def _reinvoke(instruction: str) -> str:
            revised_user = user_prompt + f"\n\nUser instruction: {instruction}\nRevise your fixes accordingly."
            return self._invoke_llm(system_prompt, revised_user)

        reinvoke_fn = _reinvoke if (system_prompt and user_prompt) else None
        return ask_user_confirmation(label, proposed_text, reinvoke_fn=reinvoke_fn)

    def patch_file(self, fname: str, block: str) -> None:
        try:
            with open(fname, "r", errors="ignore") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"[{self.__class__.__name__}] {fname} not found, cannot patch.")
            return

        original_content = content
        changed = False

        def replace_exact_block_once(
            source: str,
            target: str,
            replacement: str,
        ):
            pattern = re.compile(
                r"^" + re.escape(target) + r"(?=\r?$)",
                flags=re.MULTILINE,
            )
            updated, count = pattern.subn(
                lambda _match: replacement,
                source,
                count=1,
            )
            return updated, count == 1

        def find_action(patterns):
            for pattern in patterns:
                match = re.search(pattern, block)
                if match:
                    return match
            return None

        def code_block_text(match, group: int) -> str:
            return match.group(group).strip("\r\n")

        m = re.search(r"ACTION:\s*Overwrite entire file with:\s*```([\s\S]+?)```", block)
        if m:
            content = code_block_text(m, 1) + "\n"
            changed = True

        m = re.search(r"ACTION:\s*Append at end:\s*```([\s\S]+?)```", block)
        if m:
            add = code_block_text(m, 1)
            if not content.endswith("\n"):
                content += "\n"
            content += add + "\n"
            changed = True

        m = re.search(r"ACTION:\s*Remove the line:\s*```([\s\S]+?)```", block)
        if m:
            target = code_block_text(m, 1)
            content, replaced = replace_exact_block_once(content, target, "")
            changed = changed or replaced
            if not replaced:
                print(f"[{self.__class__.__name__}] WARNING: remove-target not found in {fname}")

        m = re.search(
            r"ACTION:\s*Replace:\s*```([\s\S]+?)```\s*with:\s*```([\s\S]+?)```",
            block,
        )
        if m:
            old_block = code_block_text(m, 1)
            new_block = code_block_text(m, 2)
            content, replaced = replace_exact_block_once(
                content,
                old_block,
                new_block,
            )
            changed = changed or replaced
            if not replaced:
                print(f"[{self.__class__.__name__}] WARNING: old_block not found in {fname}")

        m = find_action(
            (
                r"ACTION:\s*After the line:\s*```([\s\S]+?)```\s*add:\s*```([\s\S]+?)```",
                r"ACTION:[^\n]*\n\s*After the line:\s*```([\s\S]+?)```\s*add:\s*```([\s\S]+?)```",
            )
        )
        if m:
            target = code_block_text(m, 1)
            insert = code_block_text(m, 2)
            content, replaced = replace_exact_block_once(
                content,
                target,
                target + "\n" + insert,
            )
            changed = changed or replaced
            if not replaced:
                print(f"[{self.__class__.__name__}] WARNING: after-target not found in {fname}")

        m = find_action(
            (
                r"ACTION:\s*Before the line:\s*```([\s\S]+?)```\s*add:\s*```([\s\S]+?)```",
                r"ACTION:[^\n]*\n\s*Before the line:\s*```([\s\S]+?)```\s*add:\s*```([\s\S]+?)```",
            )
        )
        if m:
            target = code_block_text(m, 1)
            insert = code_block_text(m, 2)
            content, replaced = replace_exact_block_once(
                content,
                target,
                insert + "\n" + target,
            )
            changed = changed or replaced
            if not replaced:
                print(f"[{self.__class__.__name__}] WARNING: before-target not found in {fname}")

        if changed and content != original_content:
            with open(fname, "w") as f:
                f.write(content)
            print(f"{fname} has been automatically modified.")
        else:
            print(f"No applicable modifications found in {fname}.")
