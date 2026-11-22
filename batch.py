import csv
from io import StringIO

import pandas as pd
from sklearn.pipeline import Pipeline


class CSVInputError(ValueError):
    """The uploaded file cannot be classified as a message CSV."""


def classify_csv(contents: bytes, pipeline: Pipeline) -> pd.DataFrame:
    """Classify a UTF-8 CSV and preserve its original message rows."""
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CSVInputError("Save the CSV with UTF-8 encoding and upload it again.") from error

    if not text.strip():
        raise CSVInputError("The CSV file is empty.")

    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, [])
        if header.count("message") != 1:
            raise CSVInputError("The CSV must have exactly one 'message' column.")
        message_column = header.index("message")
        messages = []
        for row in reader:
            if row and len(row) != len(header):
                raise CSVInputError(
                    "A CSV row has a different number of fields than the header. "
                    "Check the column separators and put messages with commas in double quotes."
                )
            messages.append(row[message_column] if row else "")
    except csv.Error as error:
        raise CSVInputError("Upload a valid CSV file with a 'message' column.") from error

    if not messages:
        raise CSVInputError("The CSV file has no message rows.")

    cleaned = [" ".join(message.split()) for message in messages]
    nonblank_rows = [index for index, message in enumerate(cleaned) if message]
    predictions = [""] * len(messages)
    if nonblank_rows:
        labels = pipeline.predict([cleaned[index] for index in nonblank_rows])
        for index, label in zip(nonblank_rows, labels, strict=True):
            predictions[index] = label

    return pd.DataFrame({"message": messages, "prediction": predictions})


def export_predictions(results: pd.DataFrame) -> str:
    return results.to_csv(index=False, lineterminator="\r\n")
