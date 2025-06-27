from agent.query import analyze_mof_query
from ZeoPP.zeopp_agent import run_zeopp_pipeline
from config import chat_model
from langchain.schema import HumanMessage, SystemMessage

def make_natural_answer(user_query, zeopp_info, result_dict):
    """
    zeopp_info: LLM이 만든 structured dict
    result_dict: Zeo++ 결과 파일에서 파싱한 값(dict)
    """
    prompt = f"""
You are a MOF simulation expert.
Below is the user's original query:
"{user_query}"

The result file was parsed and the following values were obtained:
{result_dict}

Please provide a concise and clear answer to the user's question in natural language, including the key result values.
"""
    messages = [
        SystemMessage(content="You are a MOF simulation expert. Answer in natural language."),
        HumanMessage(content=prompt)
    ]
    response = chat_model(messages)
    return response.content

def interactive_mode():
    print("MOF Simulation Analyzer. Type 'quit' to exit.")
    print("=" * 50)
    user_input = input("\nUser input: ").strip()
    if user_input.lower() in ['quit', 'exit']:
        print("Exiting program.")
        return None
    print("\nAnalyzing...")
    query = analyze_mof_query(user_input)
    print("\n" + "=" * 50)
    print("Parsed query:", query)
    if query and query.get("simulation_tool", "").lower() == "zeopp":
        # 1. Zeo++ 실행
        result = run_zeopp_pipeline(user_input)
        answer = make_natural_answer(user_input, query, result)
        print("\n[Answer]\n", answer)
        return answer
    else:
        print("지원하지 않는 시뮬레이션 툴입니다.")
        return None

if __name__ == "__main__":
    interactive_mode()
