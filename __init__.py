from .node import RN_Translator_Node

NODE_CLASS_MAPPINGS = {
    "RN_Translator_Node": RN_Translator_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RN_Translator_Node": "RunNode 翻译器",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
