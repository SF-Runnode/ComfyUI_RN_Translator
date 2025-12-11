from .node import RN_Translator_Node, RN_Prompt_Translator, RN_SeedXPro_Translator

NODE_CLASS_MAPPINGS = {
    "RN_Translator_Node": RN_Translator_Node,
    "RN Prompt Translator": RN_Prompt_Translator,
    "RN SeedXPro Translator": RN_SeedXPro_Translator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RN_Translator_Node": "RunNode 翻译器",
    "RN Prompt Translator": "RunNode Translater",
    "RN SeedXPro Translator": "RunNode SeedXPro Translater",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
