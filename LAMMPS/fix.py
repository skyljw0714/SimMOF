from openai import OpenAI
import os

client = OpenAI(api_key="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")

def read_file(filepath, max_lines=200):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines//2] + ['\n...\n'] + lines[-max_lines//2:]
    return ''.join(lines)

def extract_lammps_error(logfile="log.lammps"):
    lines = []
    with open(logfile, "r") as f:
        for line in f:
            if "ERROR:" in line:
                lines.append(line.strip())
    return '\n'.join(lines)

def call_llm_for_fix(error_msg, file_dict):
    # file_dict: {"system.in": "...", "system.data": "...", ...}
    system_prompt = (
        "You are a LAMMPS simulation troubleshooting assistant.\n"
        "Given an ERROR message and input files, for each fix you suggest, you MUST begin by stating the filename to fix "
        "using this format:\n"
        "FILE: <filename>\n"
        "ACTION: <pattern description>\n"
        "SUGGESTED CHANGE:\n<payload>\n"
        "Use ONLY ONE of these action patterns for each fix:\n"
        "1. After the line:\n```<text>```\nadd:\n```<text to insert>```\n"
        "2. Before the line:\n```<text>```\nadd:\n```<text to insert>```\n"
        "3. Remove the line:\n```<exact line to remove>```\n"
        "4. Replace:\n```<old line(s)>```\nwith:\n```<new line(s)>```\n"
        "5. Append at end:\n```<text to append>```\n"
        "6. Overwrite entire file with:\n```<new content>```\n"
        "For EACH fix, output a separate block as above.\n"
        "If there are multiple fixes, SEPARATE EACH BLOCK by exactly four dashes `----` on a line by themselves.\n"
        "Do NOT use any other separator between blocks except `----`.\n"
        "Return your response STRICTLY as described above."
    )

    user_prompt = f"ERROR message from LAMMPS log:\n{error_msg}\n\n"
    for fname, content in file_dict.items():
        user_prompt += f"\n----- {fname} -----\n{content}\n"

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def apply_fix_to_file(fname, fix_instruction):
    with open(fname, "w") as f:
        f.write(fix_instruction)

import re

def patch_file_from_llm(fname, block):
    with open(fname, 'r') as f:
        lines = f.readlines()
    changed = False

    # 1. After the line ... add ...
    m = re.search(r'ACTION:\s*After the line:\s*```([\s\S]+?)```\s*add:\s*```([\s\S]+?)```', block)
    if m:
        target = m.group(1).strip()
        insert = m.group(2).strip()
        new_lines = []
        inserted = False
        for line in lines:
            new_lines.append(line)
            if not inserted and target in line:
                new_lines.append(insert + '\n')
                inserted = True
        lines = new_lines
        changed = inserted

    # 2. Before the line ... add ...
    m = re.search(r'ACTION:\s*Before the line:\s*```([\s\S]+?)```\s*add:\s*```([\s\S]+?)```', block)
    if m:
        target = m.group(1).strip()
        insert = m.group(2).strip()
        new_lines = []
        inserted = False
        for line in lines:
            if not inserted and target in line:
                new_lines.append(insert + '\n')
                inserted = True
            new_lines.append(line)
        lines = new_lines
        changed = inserted

    # 3. Remove the line ...
    m = re.search(r'ACTION:\s*Remove the line:\s*```([\s\S]+?)```', block)
    if m:
        target = m.group(1).strip()
        old_count = len(lines)
        lines = [line for line in lines if target not in line]
        changed = (len(lines) != old_count)

    # 4. Replace ... with ...
    m = re.search(r'ACTION:\s*Replace:\s*```([\s\S]+?)```\s*with:\s*```([\s\S]+?)```', block)
    if m:
        old = m.group(1).strip()
        new = m.group(2).strip()
        lines = [line.replace(old, new) if old in line else line for line in lines]
        changed = True

    # 5. Append at end ...
    m = re.search(r'ACTION:\s*Append at end:\s*```([\s\S]+?)```', block)
    if m:
        add = m.group(1).strip()
        if not lines or not lines[-1].endswith('\n'):
            lines.append('\n')
        lines.append(add + '\n')
        changed = True

    # 6. Overwrite entire file with ...
    m = re.search(r'ACTION:\s*Overwrite entire file with:\s*```([\s\S]+?)```', block)
    if m:
        new_content = m.group(1).strip() + '\n'
        lines = [new_content]
        changed = True

    if changed:
        with open(fname, 'w') as f:
            f.writelines(lines)
        print(f"{fname} 자동 수정 완료.")
    else:
        print(f"{fname}에서 적용할 수정사항이 없습니다.")


def main():
    err = extract_lammps_error("log.lammps")

    files = ["system.in", "system.in.settings", "system.in.init"]
    file_dict = {fname: read_file(fname) for fname in files}

    fix = call_llm_for_fix(err, file_dict)
    print("LLM SUGGESTION:\n", fix)



if __name__ == "__main__":
    main()