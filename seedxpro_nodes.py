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

def extract_translation_from_output(output, dst_code):
    patterns = [
        f'<{dst_code}>(.*?)(?:<(?!/)|$)',
        f'<{dst_code}>(.*)',
        f'{dst_code}>(.*?)(?:<|$)',
        f'<{dst_code}>\s*(.*?)(?:\n\n|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            if result:
                return result
    lines = output.split('\n')
    found_marker = False
    result_lines = []
    for line in lines:
        if f'<{dst_code}>' in line or f'{dst_code}>' in line:
            found_marker = True
            parts = re.split(f'<{dst_code}>|{dst_code}>', line, 1)
            if len(parts) > 1 and parts[1].strip():
                result_lines.append(parts[1].strip())
            continue
        if found_marker:
            if re.search(r'<[a-z]{2}>', line, re.IGNORECASE):
                break
            result_lines.append(line)
    if result_lines:
        return '\n'.join(result_lines).strip()
    return None

def translate_single_chunk(chunk, src, dst, dst_code, model, tokenizer):
    message = f"Translate from {src} to {dst}:\n{chunk}\n\nTranslation in {dst} <{dst_code}>:"
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
            translation = extract_translation_from_output(res, dst_code)
            if translation and len(translation.strip()) > 0:
                return translation
        except Exception:
            continue
    return f"[Translation failed for: {chunk}]"

def translate(**kwargs):
    prompt = kwargs.get('prompt')
    original_length = len(prompt)
    prompt = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt)
    if not prompt.strip():
        return "Error: Empty input after cleaning"
    src = kwargs.get('from')
    dst = kwargs.get('to')
    dst_code = kwargs.get('dst_code')
    model_path = os.path.join(folder_paths.models_dir, 'Seed-X-PPO-7B')
    ensure_model_downloaded(model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as model_error:
        return f'Model loading failed: {model_error}. Please delete model directory and re-download.'
    chunks = split_text_into_chunks(prompt, max_chunk_size=400)
    if len(chunks) == 1:
        result = translate_single_chunk(chunks[0], src, dst, dst_code, model, tokenizer)
    else:
        translated_chunks = []
        for chunk in chunks:
            translation = translate_single_chunk(chunk, src, dst, dst_code, model, tokenizer)
            translated_chunks.append(translation)
        result = ' '.join(translated_chunks)
    return result

class RN_SeedXPro_Translator:
    language_code_map = {
        "Arabic": "ar",
        "French": "fr",
        "Malay": "ms",
        "Russian": "ru",
        "Czech": "cs",
        "Croatian": "hr",
        "Norwegian Bokmal": "nb",
        "Swedish": "sv",
        "Danish": "da",
        "Hungarian": "hu",
        "Dutch": "nl",
        "Thai": "th",
        "German": "de",
        "Indonesian": "id",
        "Norwegian": "no",
        "Turkish": "tr",
        "English": "en",
        "Italian": "it",
        "Polish": "pl",
        "Ukrainian": "uk",
        "Spanish": "es",
        "Japanese": "ja",
        "Portuguese": "pt",
        "Vietnamese": "vi",
        "Finnish": "fi",
        "Korean": "ko",
        "Romanian": "ro",
        "Chinese": "zh"
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "may the force be with you"}),
                "from": (list(cls.language_code_map.keys()), {'default': 'English'}),
                "to": (list(cls.language_code_map.keys()), {'default': 'Chinese'}),
                "seed": ("INT", {"default": 28, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content",)
    FUNCTION = "translate"
    CATEGORY = "RN翻译/SeedXPro"

    def translate(self, **kwargs):
        kwargs['dst_code'] = self.language_code_map[kwargs.get('to')]
        res = translate(**kwargs)
        return (res,)

