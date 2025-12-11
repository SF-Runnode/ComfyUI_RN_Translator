import os
import re
import torch
import folder_paths
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

def ensure_model_downloaded(model_path, repo_id="ByteDance-Seed/Seed-X-PPO-7B"):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )

def split_text_into_chunks(text, max_chunk_size=500):
    if len(text) <= max_chunk_size:
        return [text]
    sentences = re.split(r'[.!?。！？]\s*', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_chunk_size:
            current_chunk += (sentence + ". ") if sentence else ""
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = (sentence + ". ") if sentence else ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_chunk_size):
                final_chunks.append(chunk[i:i + max_chunk_size])
    return final_chunks

def cleanup_output(output):
    text = output.strip()
    return text

def translate_single_chunk(chunk, target_lang, model, tokenizer):
    if target_lang == "English":
        instruct = "Convert the following natural language into a high-quality prompt in English for text-to-image models. Be concise and specific, use comma-separated descriptors. Do not add explanations. Output only the prompt."
    else:
        instruct = "将以下自然语言转换为中文的高质量绘图提示词。保持简洁具体，使用逗号分隔的描述，不要添加解释或多余内容。只输出提示词。"
    message = f"{instruct}\n\nInput:\n{chunk}\n\nPrompt:"
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    max_tokens = min(1024, max(150, len(chunk) * 2))
    for attempt in range(2):
        try:
            if attempt == 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned = cleanup_output(res)
            if cleaned:
                return cleaned
        except Exception:
            continue
    return f"[Prompt generation failed for: {chunk}]"

def translate_prompt(prompt, target_lang):
    model_path = os.path.join(folder_paths.models_dir, 'Seed-X-PPO-7B')
    ensure_model_downloaded(model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as model_error:
        return f'Model loading failed: {model_error}. Please delete model directory and re-download.'
    chunks = split_text_into_chunks(prompt, max_chunk_size=400)
    if len(chunks) == 1:
        result = translate_single_chunk(chunks[0], target_lang, model, tokenizer)
    else:
        translated_chunks = []
        for chunk in chunks:
            translation = translate_single_chunk(chunk, target_lang, model, tokenizer)
            translated_chunks.append(translation)
        result = ' '.join(translated_chunks)
    return result

def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

class RN_Prompt_Translator:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "may the force be with you"}),
                "seed": ("INT", {"default": 28, "min": 0, "max": 0xffffffffffffffff}),
                "advanced_options": (["auto", "force_English", "force_Chinese"], {'default': 'auto'}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translation",)
    FUNCTION = "translate"
    CATEGORY = "RN翻译/Prompt"

    def translate(self, prompt, seed, advanced_options):
        text = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt or "")
        if not text.strip():
            return ("",)
        if advanced_options == "force_English":
            target_lang = "English"
        elif advanced_options == "force_Chinese":
            target_lang = "Chinese"
        else:
            if contains_chinese(text):
                target_lang = "English"
            else:
                target_lang = "Chinese"
        res = translate_prompt(text, target_lang)
        return (res,)
