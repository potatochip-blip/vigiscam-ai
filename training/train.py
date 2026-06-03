"""Fine-tune a ViT image classifier for deepfake (real-vs-fake) detection.

This is the "train the AI" pipeline. It fine-tunes an open Vision Transformer
on YOUR labelled face dataset and produces a checkpoint the worker can use via
`VIGISCAM_AI_DEEPFAKE_IMAGE_MODEL=/path/to/checkpoint` (or an HF id after
upload). Run it on a GPU box — see training/README.md.

Dataset layout (standard ImageFolder):

    data/
      train/
        real/  *.jpg ...
        fake/  *.jpg ...
      val/
        real/  *.jpg ...
        fake/  *.jpg ...

The model is trained as binary {real, fake}; the worker's label mapping already
recognises those names. Metrics report accuracy AND the **false-negative rate**
on the fake class — the number that matters for anti-scam (a fake scored real).

Example:
    python train.py --data-dir ./data --base google/vit-base-patch16-224 \
        --output ./vigiscam-deepfake-vit --epochs 5 --batch-size 32
"""
from __future__ import annotations

import argparse
import numpy as np


def build_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        # label ids: resolved at runtime from the dataset; we compute the
        # false-negative rate for the "fake" class by id passed via closure.
        fake_id = compute_metrics.fake_id
        fake_mask = labels == fake_id
        # FN = fake samples predicted as not-fake.
        fn = int(((preds != fake_id) & fake_mask).sum())
        total_fake = int(fake_mask.sum())
        fnr = (fn / total_fake) if total_fake else 0.0
        return {"accuracy": acc, "fake_false_negative_rate": fnr}

    compute_metrics.fake_id = 1
    return compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="ImageFolder root with train/ and val/")
    ap.add_argument("--base", default="google/vit-base-patch16-224")
    ap.add_argument("--output", default="./vigiscam-deepfake-vit")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    # Imported here so the file is importable without the heavy deps installed.
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Trainer,
        TrainingArguments,
    )

    ds = load_dataset("imagefolder", data_dir=args.data_dir)
    labels = ds["train"].features["label"].names  # e.g. ['fake', 'real']
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    fake_id = next((i for l, i in label2id.items() if "fake" in l.lower()), 1)

    processor = AutoImageProcessor.from_pretrained(args.base)

    def transform(batch):
        imgs = [img.convert("RGB") for img in batch["image"]]
        batch["pixel_values"] = processor(imgs, return_tensors="pt")["pixel_values"]
        return batch

    ds = ds.with_transform(transform)

    model = AutoModelForImageClassification.from_pretrained(
        args.base,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch]),
        }

    metrics = build_metrics()
    metrics.fake_id = fake_id

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="fake_false_negative_rate",
        greater_is_better=False,  # minimise missed fakes
        logging_steps=50,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("val") or ds.get("validation") or ds["train"],
        data_collator=collate,
        compute_metrics=metrics,
    )

    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\nSaved fine-tuned model to {args.output}")
    print("Point the worker at it:")
    print(f"  VIGISCAM_AI_DEEPFAKE_IMAGE_MODEL={args.output}")


if __name__ == "__main__":
    main()
