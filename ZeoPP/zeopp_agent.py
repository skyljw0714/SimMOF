import subprocess
import os
from .prompt import ZEOPP_DESCRIPTION, ZEOPP_EXAMPLES
from config import chat_model
from langchain.schema import HumanMessage, SystemMessage
import json

working_dir = '/home/users/skyljw0714/AutoResearch/working_dir'
zeo_dir = '/home/users/skyljw0714/AutoResearch/ZeoPP'

def get_zeopp_command(zeopp_info: dict, cif_dir: str = working_dir) -> str:
    """
    zeopp_info(dict)를 받아서 Zeo++ 실행 명령어 문자열을 반환합니다.
    cif_dir: .cif 파일이 위치한 디렉토리 (기본값: 현재 디렉토리)
    """
    mof = zeopp_info["MOF"].lower()
    command = zeopp_info["command"]
    probe_diameter = zeopp_info.get("probe_diameter")
    num_samples = zeopp_info.get("num_samples")
    cif_file = os.path.join(cif_dir, f"{mof}.cif") if cif_dir else f"{mof}.cif"

    cmd = [os.path.join(zeo_dir, "network")] + command.split()
    # probe_diameter, num_samples가 None이 아니면 인자 추가
    if probe_diameter is not None:
        radius = str(probe_diameter)
        if num_samples is not None:
            cmd += [radius, radius, str(num_samples)]
        elif "block" in command or "chan" in command:
            cmd += [radius]
    elif num_samples is not None:
        cmd += [str(num_samples)]
    cmd.append(cif_file)
    return " ".join(cmd)

def run_zeopp_command(zeopp_command: str):
    """
    Run Zeo++ command using subprocess.
    """
    subprocess.run(zeopp_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_zeopp_info(raw_query: str):
    """
    Use LLM to convert a raw natural language query to a structured Zeo++ command dict.
    """
    prompt = f"""
    You are an expert in Zeo++ (zeopp) command-line usage for MOF analysis.
    {ZEOPP_DESCRIPTION}

    Below are some examples of how to convert user queries to Zeo++ command parameters:
    {ZEOPP_EXAMPLES}

    Now, given the following user query, extract the necessary parameters and generate a Zeo++ command and arguments in structured JSON format.
    User query: "{raw_query}"

    Return only the JSON object as shown in the examples above.
    """
    
    messages = [
        SystemMessage(content="You are a Zeo++ command expert. Output only the JSON object."),
        HumanMessage(content=prompt)
    ]
    response = chat_model(messages)
    
    zeopp_info = json.loads(response.content)
    
    try:
        return zeopp_info
    except Exception as e:
        print("LLM parsing error:", e)
        print("Raw response:", response.content)
        return None


def read_res_file(mof: str, working_dir: str = ".") -> dict:
    """
    {mof}.res 파일을 읽어 pore diameters 값을 반환합니다.
    Returns: {"included_sphere": float, "free_sphere": float, "included_sphere_along_free_path": float}
    """
    res_path = os.path.join(working_dir, f"{mof}.res")
    with open(res_path, "r") as f:
        line = f.readline().strip()
        # 파일 형식: {mof}.res  4.89082 3.03868  4.81969
        parts = line.split()
        return {
            "included_sphere": float(parts[1]),
            "free_sphere": float(parts[2]),
            "included_sphere_along_free_path": float(parts[3])
        }
    return {}

def read_vol_file(mof: str, working_dir: str = ".") -> dict:
    """
    {mof}.vol 파일을 읽어 accessible volume 값을 반환합니다.
    Returns: {"AV_A3": float, "AV_Volume_fraction": float, "AV_cm3_g": float}
    """
    vol_path = os.path.join(working_dir, f"{mof}.vol")
    with open(vol_path, "r") as f:
        for line in f:
            if line.startswith("@"):
                # 파일 형식: @ {mof}.vol Unitcell_volume: {vol} Density: {density} AV_A^3: {av} AV_Volume_fraction: {vf} AV_cm^3/g: {cm3}
                values = line.strip().split()
                #print(values)
                return {
                    "AV_A3": float(values[7]),
                    "AV_Volume_fraction": float(values[9]),
                    "AV_cm3_g": float(values[11])
                }
    return {}

def read_sa_file(mof: str, working_dir: str = ".") -> dict:
    """
    {mof}.sa 파일을 읽어 accessible surface area 값을 반환합니다.
    Returns: {"ASA_A2": float, "ASA_m2_cm3": float, "ASA_m2_g": float}
    """
    sa_path = os.path.join(working_dir, f"{mof}.sa")
    with open(sa_path, "r") as f:
        for line in f:
            if line.startswith("@"):
                values =line.strip().split()
                print(values)
                return {
                    "ASA_A2": float(values[7]),
                    "ASA_m2_cm3": float(values[9]),
                    "ASA_m2_g": float(values[11])
                }
    return {}

def run_zeopp_pipeline(user_query: str, working_dir: str = working_dir) -> dict:
    """
    user_query를 받아 Zeo++ 전체 파이프라인(LLM 분석→명령어 생성→실행→결과 파싱→결과 dict 반환)
    """
    # 1. LLM으로 zeopp_info 생성
    zeopp_info = get_zeopp_info(user_query)
    if not zeopp_info:
        print("[ERROR] LLM 분석 실패")
        return {}
    # 2. 명령어 생성
    zeopp_command = get_zeopp_command(zeopp_info, cif_dir=working_dir)
    print(f"[INFO] Zeo++ 명령어: {zeopp_command}")
    # 3. 명령어 실행
    run_zeopp_command(zeopp_command)
    # 4. 결과 파일 파싱 (명령어 종류에 따라)
    mof = zeopp_info["MOF"].lower()
    command = zeopp_info["command"]
    if "-res" in command:
        result = read_res_file(mof, working_dir)
    elif "-vol" in command:
        result = read_vol_file(mof, working_dir)
    elif "-sa" in command:  
        result = read_sa_file(mof, working_dir)
    else:
        result = {}
    return {
        "zeopp_info": zeopp_info,
        "zeopp_command": zeopp_command,
        "result": result
    }
