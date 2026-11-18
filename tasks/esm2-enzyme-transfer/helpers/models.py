"""Embedding, fine-tuning, and prediction helpers built on ESM-2 and Hugging Face."""

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from helpers.data import CLASS_NAMES

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
N_CLASSES = len(CLASS_NAMES)
# Batches are padded to a multiple of this many tokens. Few distinct tensor shapes keep the
# MPS (Apple Silicon) allocator cache from growing without bound; 1024 is a multiple of it.
PAD_MULTIPLE = 64


def pick_device() -> str:
    """Best available device: 'cuda', else 'mps' (Apple Silicon), else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def _quiet_loading():
    """Silence the 'weights not initialized / unused' notice for one load call."""
    previous = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers.logging.set_verbosity(previous)


def _load(auto_class, model_name_or_path, device, **kwargs):
    """Load a tokenizer and model, move the model to the device, set eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with _quiet_loading():
        model = auto_class.from_pretrained(model_name_or_path, **kwargs)
    return tokenizer, model.to(device or pick_device()).eval()


def _length_sorted_batches(sequences, batch_size):
    """Index arrays that group sequences of similar length together."""
    order = np.argsort([len(s) for s in sequences], kind="stable")
    return np.array_split(order, np.ceil(len(order) / batch_size))


def _run_batches(sequences, tokenizer, model, batch_size, max_length, summarize):
    """Run the model over length-sorted batches; return summarize(output, batch) rows in input order."""
    sequences = list(sequences)
    if not sequences:
        return None
    pieces = []
    for idx in _length_sorted_batches(sequences, batch_size):
        batch = tokenizer(
            [sequences[i] for i in idx],
            padding=True,
            pad_to_multiple_of=PAD_MULTIPLE,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = batch.pop("special_tokens_mask").to(model.device)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = model(**batch)
        pieces.append(summarize(out, batch, special).float().cpu().numpy())
    sorted_rows = np.concatenate(pieces)
    restored = np.empty_like(sorted_rows)
    restored[np.concatenate(_length_sorted_batches(sequences, batch_size))] = sorted_rows
    return restored


def embed_sequences(
    sequences: Sequence[str],
    model_name_or_path: str = MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    """Mean of the final hidden states over residue tokens, as float32 (n, hidden)."""
    tokenizer, model = _load(AutoModel, model_name_or_path, device, add_pooling_layer=False)

    def mean_pool(out, batch, special):
        keep = (batch["attention_mask"].bool() & ~special.bool()).unsqueeze(-1).to(out.last_hidden_state.dtype)
        total = (out.last_hidden_state * keep).sum(dim=1)
        return total / keep.sum(dim=1).clamp(min=1)

    result = _run_batches(sequences, tokenizer, model, batch_size, max_length, mean_pool)
    if result is None:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return result.astype(np.float32)


def cached_embeddings(
    df: pd.DataFrame, cache_path: Path, model_name_or_path: str = MODEL_NAME, **embed_kwargs
) -> np.ndarray:
    """Embeddings for df.sequence, cached in an npz keyed by accession order and model."""
    cache_path = Path(cache_path)
    accessions = df["accession"].to_numpy().astype(str)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            if str(cached["model"]) == str(model_name_or_path) and np.array_equal(cached["accessions"], accessions):
                return cached["embeddings"]
    embeddings = embed_sequences(list(df["sequence"]), model_name_or_path, **embed_kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, accessions=accessions, embeddings=embeddings, model=str(model_name_or_path))
    return embeddings


class _SequenceDataset(torch.utils.data.Dataset):
    """Tokenized sequences with zero-based integer labels."""

    def __init__(self, df, tokenizer, max_length):
        encoded = tokenizer(list(df["sequence"]), truncation=True, max_length=max_length)
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = (df["ec_class"].to_numpy() - 1).astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": int(self.labels[i]),
        }


class _WeightedTrainer(Trainer):
    """Trainer whose cross-entropy loss uses per-class weights."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _compute_metrics(eval_pred):
    pred = np.argmax(eval_pred.predictions, axis=-1)
    return {"macro_f1": f1_score(eval_pred.label_ids, pred, average="macro", zero_division=0)}


def make_trainer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    model_name: str = MODEL_NAME,
    num_epochs: int = 5,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    max_length: int = 1024,
    class_weighted: bool = True,
    seed: int = 0,
) -> Trainer:
    """Build a Trainer that fine-tunes ESM-2 for 7-class EC prediction."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with _quiet_loading():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=N_CLASSES,
            id2label={k - 1: v for k, v in CLASS_NAMES.items()},
            label2id={v: k - 1 for k, v in CLASS_NAMES.items()},
        )
    train_set = _SequenceDataset(train_df, tokenizer, max_length)
    val_set = _SequenceDataset(val_df, tokenizer, max_length)

    weights = None
    if class_weighted:
        present = np.unique(train_set.labels)
        weights = np.ones(N_CLASSES)
        weights[present] = compute_class_weight("balanced", classes=present, y=train_set.labels)
        weights = torch.tensor(weights, dtype=torch.float32)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        train_sampling_strategy="group_by_length",  # transformers v5 name for group_by_length=True
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        logging_steps=10,
        seed=seed,
    )
    return _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=PAD_MULTIPLE),
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
        class_weights=weights,
    )


def predict_classes(
    model_path: str | Path,
    sequences: Sequence[str],
    batch_size: int = 16,
    max_length: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    """Predicted EC class (1-7) for each sequence from a saved fine-tuned model."""
    tokenizer, model = _load(AutoModelForSequenceClassification, str(model_path), device)
    result = _run_batches(
        sequences, tokenizer, model, batch_size, max_length, lambda out, batch, special: out.logits
    )
    if result is None:
        return np.zeros(0, dtype=int)
    return result.argmax(axis=1) + 1


def loss_history_frame(trainer: Trainer) -> pd.DataFrame:
    """Training and evaluation log rows as columns step, epoch, train_loss, eval_loss, eval_macro_f1."""
    log = pd.DataFrame(trainer.state.log_history)
    for col in ("loss", "eval_loss", "eval_macro_f1"):
        if col not in log:
            log[col] = np.nan
    log = log[log["loss"].notna() | log["eval_loss"].notna()]
    return log.drop(columns=["train_loss"], errors="ignore").rename(columns={"loss": "train_loss"})[
        ["step", "epoch", "train_loss", "eval_loss", "eval_macro_f1"]
    ].reset_index(drop=True)
