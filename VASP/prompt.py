PARALLEL_INPUT = """
NCORE    =   4
LPLANE   =   .TRUE.
LREAL    =   Auto
IALGO    =   38
"""

VASP_FORMAT = '''
System = {system}

GGA = PE
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400
EDIFF = 1E-6
EDIFFG = -0.01
IBRION = 1
ISIF = 3 
NSW = 500
LREAL = Auto
KGAMMA = .TRUE. #Never change 

{PARALLEL_INPUT}
'''


VASP_INPUT_PROMPT = '''
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
'''
