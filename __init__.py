from .node import RN_Translator_Node
from .seedxpro_nodes import RN_SeedXPro_Translator
from .prompter import RN_Prompt_Translator

NODE_CLASS_MAPPINGS = {
    "RN_Translator_Node": RN_Translator_Node,
    "RN SeedXPro Translator": RN_SeedXPro_Translator,
    "RN Prompt Translator": RN_Prompt_Translator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RN_Translator_Node": "RN 翻译器",
    "RN SeedXPro Translator": "RN SeedXPro 翻译器",
    "RN Prompt Translator": "RN Prompt 翻译器",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
