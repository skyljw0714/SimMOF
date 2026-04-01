

class BaseErrorAgent:
    def extract_error(self):
        raise NotImplementedError

    def call_llm_for_fix(self, err_text, file_dict):
        raise NotImplementedError

    def patch_file(self, fname, patch_block):
        raise NotImplementedError


class ErrorAgentFactory:
    
    @staticmethod
    def get(agent_name: str, **kwargs):

        if agent_name == "LAMMPSErrorAgent":
            from .lammps_error import LammpsErrorAgent
            return LammpsErrorAgent(**kwargs)

        elif agent_name == "VASPErrorAgent":
            from .vasp_error import VaspErrorAgent
            return VaspErrorAgent(**kwargs)

        elif agent_name == "RASPAErrorAgent":
            from .raspa_error import RaspaErrorAgent
            return RaspaErrorAgent(**kwargs)

        else:
            raise ValueError(f"Unknown ErrorAgent type: {agent_name}")
