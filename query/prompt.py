import json
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List


SIMULATION_TOOL_DESCRIPTIONS = {
    "VASP": "VASP is a quantum mechanical simulation package based on density functional theory (DFT), mainly used for electronic structure calculations, structure optimization, band structure, and binding energy calculations.",
    "RASPA": "RASPA is a Monte Carlo simulation package for studying gas adsorption, diffusion, and mixture separation in porous materials such as MOFs (Metal-Organic Frameworks).",
    "zeopp": "zeo++ (zeopp) is a structure analysis tool for calculating surface area, pore volume, and pore size distribution of porous materials.",
    "LAMMPS": "LAMMPS is a molecular dynamics simulation package for studying the dynamics of atoms and molecules in porous materials.",
    "ScreeningAgent": "ScreeningAgent performs fast, low-cost pre-screening of large material datasets using simple structural checks, heuristic rules, or approximate property estimators to select suitable candidates for subsequent detailed simulations."
}


class VASPWorkflowStep(BaseModel):
    system: str = Field(description="System name (e.g., UiO-66, CO2, UiO-66-CO2)")
    step: int = Field(description="Step number (1 for parallel, 2+ for sequential)")
    property: str = Field(description="What to calculate (e.g., energy, optimization, DOS, bader charge)")

class VASPWorkflow(BaseModel):
    workflow: List[VASPWorkflowStep] = Field(description="List of VASP calculation steps")


class MOFSimulation(BaseModel):
    simulation_tool: str = Field(description="Simulation tool (e.g., VASP, RASPA, GROMACS)")
    simulation_property: str = Field(description="Property to calculate (e.g., binding energy, gas uptake, diffusion coefficient)")
    MOF: str = Field(description="MOF structure name (e.g., HKUST-1, UiO-66, ZIF-8)")
    guest: Optional[str] = Field(description="Guest molecule (e.g., CO2, H2, CH4, N2)")


parser = PydanticOutputParser(pydantic_object=MOFSimulation)
vasp_workflow_parser = PydanticOutputParser(pydantic_object=VASPWorkflow)


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

def create_mof_analysis_prompt(user_input: str):
        
    
    tool_desc_text = "=== Simulation Software Descriptions ===\n"
    for tool, desc in SIMULATION_TOOL_DESCRIPTIONS.items():
        tool_desc_text += f"- {tool}: {desc}\n"
    tool_desc_text += "\n"

    
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
    
    return structured_prompt



def get_system_message():
        return "You are a MOF simulation expert. Please provide analysis results in the exact format specified."
