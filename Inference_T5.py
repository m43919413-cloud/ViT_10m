from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

ADAPTER_PATH = "./T5_model"
BASE_MODEL   = "t5-small" 

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

model = PeftModel.from_pretrained(base, ADAPTER_PATH).eval()

def summarize(dialogue):
    prompt = "summarize: " + dialogue
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        length_penalty=1.0,
        early_stopping=True,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


dialogue = """Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)"""

print(summarize(dialogue))
