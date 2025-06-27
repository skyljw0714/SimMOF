import json
from typing import List, Optional
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config import chat_model

# Simulation tool descriptions (in English)
SIMULATION_TOOL_DESCRIPTIONS = {
    "VASP": "VASP is a quantum mechanical simulation package based on density functional theory (DFT), mainly used for electronic structure calculations, structure optimization, band structure, and binding energy calculations.",
    "RASPA": "RASPA is a Monte Carlo simulation package for studying gas adsorption, diffusion, and mixture separation in porous materials such as MOFs (Metal-Organic Frameworks).",
    "zeopp": "zeo++ (zeopp) is a structure analysis tool for calculating surface area, pore volume, and pore size distribution of porous materials.",
    "LAMMPS": "LAMMPS is a molecular dynamics simulation package for studying the dynamics of atoms and molecules in porous materials."
}

# Pydantic model definition
class MOFSimulation(BaseModel):
    simulation_tool: str = Field(description="Simulation tool (e.g., VASP, RASPA, GROMACS)")
    simulation_property: str = Field(description="Property to calculate (e.g., binding energy, gas uptake, diffusion coefficient)")
    MOF: str = Field(description="MOF structure name (e.g., HKUST-1, UiO-66, ZIF-8)")
    guest: Optional[str] = Field(description="Guest molecule (e.g., CO2, H2, CH4, N2)")

# Pydantic parser creation
parser = PydanticOutputParser(pydantic_object=MOFSimulation)

# Few-shot examples
examples = [
    {
        "input": "I want to reproduce Binding energy of CO2 in HKUST-1 MOF",
        "output": {
            "simulation_tool": "VASP",
            "simulation_property": "binding energy",
            "MOF": "HKUST-1",
            "guest": "CO2"
        }
    },
    {
        "input": "what is H2 uptake value of UiO-66 in 273K",
        "output": {
            "simulation_tool": "RASPA",
            "simulation_property": "gas uptake",
            "MOF": "UiO-66",
            "guest": "H2"
        }
    },
    {
        "input": "surface area of MOF-5",
        "output": {
            "simulation_tool": "zeopp",
            "simulation_property": "surface area",
            "MOF": "MOF-5",
            "guest": None
        }
        
    }
]

def analyze_mof_query(user_input: str):
    """Analyze user input and return structured MOF simulation information"""
    
    # Add simulation tool descriptions to the prompt
    tool_desc_text = "=== Simulation Software Descriptions ===\n"
    for tool, desc in SIMULATION_TOOL_DESCRIPTIONS.items():
        tool_desc_text += f"- {tool}: {desc}\n"
    tool_desc_text += "\n"

    # Create prompt with examples
    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Input: {example['input']}\n"
        examples_text += f"Output: {json.dumps(example['output'], ensure_ascii=False, indent=2)}\n\n"
    
    structured_prompt = f"""
You are a MOF (Metal-Organic Framework) simulation expert.
Below are descriptions of each simulation software.

{tool_desc_text}

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
        print(f"2) simulation property: {structured_response.simulation_property}")
        print(f"3) MOF: {structured_response.MOF}")
        print(f"4) guest: {structured_response.guest}")
        
        # Also output in JSON format
        print(f"\n=== JSON Format ===")
        print(json.dumps(structured_response.dict(), ensure_ascii=False, indent=2))
        
        query_result = structured_response.dict()
        query_result['raw_query'] = user_input
        
        return query_result
        
    except Exception as e:
        print(f"Parsing error: {e}")
        print("Original response:", response.content)
        return None
