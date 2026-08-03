from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public" / "model"
REPORT_DIR = ROOT / "reports"
SEED = 42
MAX_FEATURES = 2000
CLASSES = [
    "invoice",
    "receipt",
    "bank_statement",
    "purchase_order",
    "payslip",
    "tax_form",
    "insurance_claim",
    "financial_report",
    "unknown",
]

COMPANIES = ["Northwind Analytics", "Atlas Retail", "Acme AI", "Cedar Finance", "Blue Harbor LLC", "Nova Health"]
NAMES = ["Maya Chen", "Hani Ghena", "Adam Bell", "Nora Haddad", "Leila Mansour", "Omar Stone"]
ITEMS = ["data audit", "cloud hosting", "consulting hours", "office supplies", "subscription", "hardware kit"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9@._+#$%-]+", text.lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return {word: index for index, (word, _) in enumerate(counts.most_common(MAX_FEATURES))}


def vectorize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            index = vocab.get(token)
            if index is not None:
                matrix[row, index] += 1.0
        matrix[row] = np.log1p(matrix[row])
    return matrix


def money() -> str:
    return random.choice(["$", "USD ", "EUR "]) + f"{random.randint(12, 12000):,}.{random.randint(0, 99):02d}"


def date() -> str:
    month = random.choice(["January", "February", "March", "April", "May", "June", "July", "August", "September"])
    return f"{month} {random.randint(1, 28)}, 2026"


def iban() -> str:
    return f"DZ{random.randint(10,99)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"


def examples_for(label: str, n: int) -> list[str]:
    rows: list[str] = []
    for i in range(n):
        company = random.choice(COMPANIES)
        other = random.choice([c for c in COMPANIES if c != company])
        person = random.choice(NAMES)
        if label == "invoice":
            rows.append(
                f"Invoice INV-{2026}-{1000+i} from {company} to {other}. Issue date {date()}. Due date {date()}. "
                f"Line items: {random.choice(ITEMS)} at {money()}, {random.choice(ITEMS)} at {money()}. Tax {money()}. Total due {money()}."
            )
        elif label == "receipt":
            rows.append(
                f"Receipt #{random.randint(100000,999999)} merchant {company}. Paid by card ending {random.randint(1000,9999)} on {date()}. "
                f"Items purchased: {random.choice(ITEMS)}, {random.choice(ITEMS)}. Amount paid {money()} including VAT."
            )
        elif label == "bank_statement":
            rows.append(
                f"Bank statement for account {iban()} holder {person}. Statement period {date()} to {date()}. "
                f"Opening balance {money()}, closing balance {money()}. Transactions: salary deposit, ATM withdrawal, transfer fee."
            )
        elif label == "purchase_order":
            rows.append(
                f"Purchase Order PO-{random.randint(10000,99999)} buyer {company} supplier {other}. Requested delivery {date()}. "
                f"Approved budget {money()}. Quantity {random.randint(1,50)} units. Payment terms net {random.choice([15,30,45])}."
            )
        elif label == "payslip":
            rows.append(
                f"Payslip for employee {person}. Employer {company}. Pay period {date()}. Gross salary {money()}, deductions {money()}, "
                f"social security contribution {money()}, net pay {money()}."
            )
        elif label == "tax_form":
            rows.append(
                f"Tax form for {person} taxpayer id TAX-{random.randint(100000,999999)}. Fiscal year 2026. "
                f"Declared income {money()}, deductible expenses {money()}, tax withheld {money()}, refund or balance due {money()}."
            )
        elif label == "insurance_claim":
            rows.append(
                f"Insurance claim CLM-{random.randint(10000,99999)} policy POL-{random.randint(10000,99999)} claimant {person}. "
                f"Incident date {date()}. Claimed amount {money()}. Adjuster notes, deductible, coverage review, reimbursement decision."
            )
        elif label == "financial_report":
            rows.append(
                f"{company} quarterly financial report. Revenue {money()}, operating expenses {money()}, EBITDA {money()}, "
                f"net income {money()}, cash flow, balance sheet notes, forecast, and management commentary."
            )
        else:
            rows.append(
                random.choice(
                    [
                        "Meeting notes about design decisions, colors, layout, and product roadmap.",
                        "Personal journal entry about studying machine learning and planning the week.",
                        "Recipe instructions with tomatoes, onions, oil, spices, and cooking time.",
                        "Travel plan with hotel details, weather, museum visits, and local food.",
                    ]
                )
            )
    return rows


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    texts: list[str] = []
    labels: list[int] = []
    for label_index, label in enumerate(CLASSES):
        count = 560 if label != "unknown" else 780
        generated = examples_for(label, count)
        texts.extend(generated)
        labels.extend([label_index] * len(generated))

    y = np.array(labels)
    train_texts, test_texts, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=SEED, stratify=y)
    train_texts, val_texts, y_train, y_val = train_test_split(
        train_texts, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    vocab = build_vocab(train_texts)
    x_train = vectorize(train_texts, vocab)
    x_val = vectorize(val_texts, vocab)
    x_test = vectorize(test_texts, vocab)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(vocab),), name="financial_bow"),
            tf.keras.layers.Dense(160, activation="relu", name="financial_features"),
            tf.keras.layers.Dropout(0.25, name="dropout"),
            tf.keras.layers.Dense(72, activation="relu", name="document_embedding"),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="document_type"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=14,
        batch_size=64,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)],
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "dataset": "Synthetic financial document routing corpus",
        "dataset_note": "Generated labeled financial texts used to train a deployable Keras document router with an unknown class.",
        "train_size": len(train_texts),
        "validation_size": len(val_texts),
        "test_size": len(test_texts),
        "classes": CLASSES,
        "vocabulary_size": len(vocab),
        "architecture": ["Dense(160, relu)", "Dropout(0.25)", "Dense(72, relu)", f"Dense({len(CLASSES)}, softmax)"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "epochs_ran": len(history.history["loss"]),
    }

    weights = model.get_weights()
    export = {
        "model_type": "keras_dense_financial_document_router",
        "classes": CLASSES,
        "unknown_threshold": 0.56,
        "vocabulary": vocab,
        "weights": {
            "dense1_kernel": weights[0].tolist(),
            "dense1_bias": weights[1].tolist(),
            "dense2_kernel": weights[2].tolist(),
            "dense2_bias": weights[3].tolist(),
            "out_kernel": weights[4].tolist(),
            "out_bias": weights[5].tolist(),
        },
        "metrics": metrics,
    }

    model.save(MODEL_DIR / "keras_financial_router.keras")
    (MODEL_DIR / "financial_router.json").write_text(json.dumps(export), encoding="utf-8")
    (REPORT_DIR / "keras_financial_router_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=CLASSES),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
