import evaluate
import numpy as np
import pandas as pd
from torch.optim import AdamW, lr_scheduler
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

metric = evaluate.load("accuracy")


def tokenize_dataset(data):
    return tokenizer(
        data["text"], max_length=416, truncation=True, padding="max_length"
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    MODEL_NAME = "airesearch/wangchanBERTa-base-att-spm-uncased"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    labels = {
        "pos": 0,
        "neu": 1,
        "neg": 2,
        "q": 3,
    }

    df = pd.read_csv("data/train.csv")
    df["label"] = df["label"].map(labels)

    np.random.seed(112)
    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )
    hg_train_data = Dataset.from_pandas(df_train)
    hg_eval_data = Dataset.from_pandas(df_val)
    hg_test_data = Dataset.from_pandas(df_test)

    train_dataset = hg_train_data.map(tokenize_dataset)
    eval_dataset = hg_eval_data.map(tokenize_dataset)
    test_dataset = hg_test_data.map(tokenize_dataset)

    train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
    eval_dataset = eval_dataset.remove_columns(["text", "__index_level_0__"])
    test_dataset = test_dataset.remove_columns(["text", "__index_level_0__"])

    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")

    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    training_args = TrainingArguments(
        output_dir="test_trainer", 
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()
