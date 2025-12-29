from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,  # Changed to Seq2SeqTrainingArguments
    Seq2SeqTrainer,  # Changed to Seq2SeqTrainer for correct saving/config
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from data import TranslationDataset
import torch
import os
import gc

torch.cuda.empty_cache()


def train():
    model_id = "google/mt5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data_base_path = os.path.join("data")
    # chinese data paths
    train_data_path = os.path.join(data_base_path, "en_yue_train.json")
    eval_data_path = os.path.join(data_base_path, "en_yue_val.json")

    # modified: Changed "en"/"zh" to "English"/"Mandarin"
    train_dataset = TranslationDataset(
        train_data_path,
        tokenizer,
        src_lang="English",
        tgt_lang="Cantonese",
        max_length=128,
    )
    eval_dataset = TranslationDataset(
        eval_data_path,
        tokenizer,
        src_lang="English",
        tgt_lang="Cantonese",
        max_length=128,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=64,
        lora_alpha=128,
        # reverted to only q/v to save memory
        target_modules=["q", "v"],
        lora_dropout=0.05,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="peft_training",
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,  # unstable for T5
        bf16=True,  # stable & fast
        gradient_checkpointing=True,  # critical memory saver
        optim="adafactor",  # uses less memory than AdamW

        dataloader_num_workers=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        logging_dir="./logs",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # gradients for checkpointing compatibility
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    torch.cuda.empty_cache()
    gc.collect()

    print("Starting training (Memory Optimized: Q/V only, Adafactor, BF16)...")
    trainer.train()

    model_path = os.path.join("model")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Training complete: model saved to {model_path}.")


if __name__ == "__main__":
    train()
