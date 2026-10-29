from google.colab import drive
drive.mount('/content/drive')

import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

DRIVE_PATH = "/content/drive/MyDrive/t5-lora-summarizer"
os.makedirs(DRIVE_PATH, exist_ok=True)

model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("knkarthick/samsum")

def preprocess(batch):
    inputs = ["summarize: " + t for t in batch["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(batch["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

args = Seq2SeqTrainingArguments(
    output_dir=DRIVE_PATH,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)
trainer.train()

trainer.save_model(DRIVE_PATH)
model.save_pretrained(DRIVE_PATH)
tokenizer.save_pretrained(DRIVE_PATH)
print(f"Model saved in: {DRIVE_PATH}")