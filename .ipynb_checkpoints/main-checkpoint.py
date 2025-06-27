import json
from typing import List, Optional
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config import chat_model

# Pydantic model definition
class MOFSimulation(BaseModel):
    simulation_tool: str = Field(description="Simulation tool (e.g., VASP, RASPA, GROMACS)")
    property: str = Field(description="Property to calculate (e.g., binding energy, gas uptake, diffusion coefficient)")
    mof: str = Field(description="MOF structure name (e.g., HKUST-1, UiO-66, ZIF-8)")
    guest: str = Field(description="Guest molecule (e.g., CO2, H2, CH4, N2)")

# Pydantic parser creation
parser = PydanticOutputParser(pydantic_object=MOFSimulation)

# Few-shot examples
examples = [
    {
        "input": "I want to reproduce Binding energy of CO2 in HKUST-1 MOF",
        "output": {
            "simulation_tool": "VASP",
            "property": "binding energy",
            "mof": "HKUST-1",
            "guest": "CO2"
        }
    },
    {
        "input": "what is H2 uptake value of UiO-66 in 273K",
        "output": {
            "simulation_tool": "RASPA",
            "property": "gas uptake",
            "mof": "UiO-66",
            "guest": "H2"
        }
    }
]

def analyze_mof_query(user_input: str):
    """Analyze user input and return structured MOF simulation information"""
    
    # Create prompt with examples
    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Input: {example['input']}\n"
        examples_text += f"Output: {json.dumps(example['output'], ensure_ascii=False, indent=2)}\n\n"
    
    structured_prompt = f"""
You are a MOF (Metal-Organic Framework) simulation expert.
Analyze the user's input and structure the simulation information.

{parser.get_format_instructions()}

{examples_text}

User input: "{user_input}"

Please provide analysis results in the exact format based on the examples above.
"""

    # Define messages
    messages = [
        SystemMessage(content="You are a MOF simulation expert. Please provide analysis results in the exact format specified."),
        HumanMessage(content=structured_prompt)
    ]

    # Generate response
    response = chat_model(messages)

    try:
        # Parse response with Pydantic parser
        structured_response = parser.parse(response.content)
        
        # Output results
        print("=== MOF Simulation Analysis Results ===")
        print(f"1) simulation tool: {structured_response.simulation_tool}")
        print(f"2) property: {structured_response.property}")
        print(f"3) MOF: {structured_response.mof}")
        print(f"4) guest: {structured_response.guest}")
        
        # Also output in JSON format
        print(f"\n=== JSON Format ===")
        print(json.dumps(structured_response.dict(), ensure_ascii=False, indent=2))
        
        return structured_response
        
    except Exception as e:
        print(f"Parsing error: {e}")
        print("Original response:", response.content)
        return None

# Interactive interface
def interactive_mode():
    """Interactive mode to receive user input and analyze"""
    print("MOF Simulation Analyzer. Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        user_input = input("\nUser input: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting program.")
            break
            
        if not user_input:
            continue
            
        print("\nAnalyzing...")
        result = analyze_mof_query(user_input)
        print("\n" + "=" * 50)

if __name__ == "__main__":
    # Run interactive mode
    interactive_mode()