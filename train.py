import json
from io import TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sms-spam-collection.zip"
MODEL_PATH = ROOT / "artifacts" / "sms-pipeline.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.json"
LABELS = ["ham", "spam"]
RANDOM_SEED = 42


def load_messages(path: Path) -> pd.DataFrame:
    """Read the UCI file from its ZIP and normalize whitespace."""
    records = []
    with ZipFile(path) as archive:
        with TextIOWrapper(archive.open("SMSSpamCollection"), encoding="utf-8-sig") as lines:
            for number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                label, separator, message = line.partition("\t")
                message = " ".join(message.split())
                if not separator or label not in LABELS or not message:
                    raise ValueError(f"Invalid SMS record on line {number}.")
                records.append((label, message))
    return pd.DataFrame(records, columns=["label", "message"])


def remove_duplicates(messages: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per message and drop messages with conflicting labels."""
    keyed = messages.assign(message_key=messages["message"].str.casefold())
    consistent = keyed.groupby("message_key")["label"].transform("nunique").eq(1)
    unique = keyed.loc[consistent].drop_duplicates(subset="message_key")
    return unique.drop(columns="message_key").reset_index(drop=True)


def evaluate(actual, predicted) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", pos_label="spam", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "spam_precision": round(float(precision), 4),
        "spam_recall": round(float(recall), 4),
        "spam_f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit("Download the UCI ZIP to data/sms-spam-collection.zip. See README.md.")

    raw = load_messages(DATA_PATH)
    messages = remove_duplicates(raw)
    train_text, test_text, train_labels, test_labels = train_test_split(
        messages["message"],
        messages["label"],
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=messages["label"],
    )

    baseline = DummyClassifier(strategy="most_frequent").fit(train_text, train_labels)
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    pipeline.fit(train_text, train_labels)

    report = {
        "raw_rows": len(raw),
        "unique_rows": len(messages),
        "train_rows": len(train_text),
        "test_rows": len(test_text),
        "test_spam_rows": int(test_labels.eq("spam").sum()),
        "test_fraction": 0.20,
        "random_seed": RANDOM_SEED,
        "confusion_matrix_labels": LABELS,
        "majority_baseline": evaluate(test_labels, baseline.predict(test_text)),
        "tfidf_logistic_regression": evaluate(test_labels, pipeline.predict(test_text)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MODEL_PATH.parent) as temporary:
        pending_model = Path(temporary) / MODEL_PATH.name
        joblib.dump(pipeline, pending_model, compress=3)
        pending_model.replace(MODEL_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2)
    REPORT_PATH.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
