from config import chat_model
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from .prompt import VASP_FORMAT, PARALLEL_INPUT
import json

def make_vasp_input_example(query: dict, PARALLEL_INPUT: str) -> str:
    """
    VASP_FORMAT 템플릿에 query와 parallelization을 채워 예시 입력 파일을 생성
    """
    prompt_template = PromptTemplate(
        input_variables=["systems", "PARALLEL_INPUT"],
        template=VASP_FORMAT
    )
    return prompt_template.format(
        system=query.get("MOF", "MOF"),
        PARALLEL_INPUT=PARALLEL_INPUT
    )

def vasp_agent(query: dict, INPUT_FOR_PARALLEL: str = "", method_paragraph: str = None) -> str:
    """
    user query(dict)와 VASP_FORMAT을 활용해 LLM을 통해 VASP 입력 파일을 생성합니다.
    INPUT_FOR_PARALLEL: 병렬화 관련 설정 문자열 (예: 'NCORE = 4\\nKPAR = 2')
    Returns: VASP 입력 파일 내용(str)
    """
    vasp_input_example = make_vasp_input_example(query, PARALLEL_INPUT)
    
    prompt = f"""
    You are a VASP input file generation expert for MOF simulations.
    Below is the standard VASP input file format:

    {vasp_input_example}
    """
    
    if method_paragraph:
        prompt += f"""
Below is a method paragraph from a scientific paper describing the simulation details. 
Please refer to this paragraph and reflect the relevant settings in the VASP INCAR file if appropriate.

Method paragraph:
{method_paragraph}
"""
    prompt += f"""
Given the following simulation request, generate a complete VASP INCAR file.

    Simulation request:
    {json.dumps(query, indent=2)}

    Please output only the VASP input file content, without any explanation.
    """
    messages = [
        SystemMessage(content="You are a VASP input file expert. Output only the VASP INCAR file content."),
        HumanMessage(content=prompt)
    ]
    response = chat_model(messages)
    return response.content
