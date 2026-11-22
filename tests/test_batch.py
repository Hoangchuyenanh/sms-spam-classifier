import csv
import unittest
from io import StringIO

from batch import CSVInputError, classify_csv, export_predictions


class RecordingPipeline:
    def __init__(self):
        self.calls = []

    def predict(self, messages):
        self.calls.append(list(messages))
        return [{"00123": "ham", 'Win, "now" please': "spam"}[message] for message in messages]


class BatchClassificationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = RecordingPipeline()

    def test_preserves_original_rows_and_predicts_normalized_messages_in_one_call(self):
        contents = (
            '\ufeffid,message\r\n'
            '4,00123\r\n'
            '3,"Win, ""now""\rplease"\r\n'
            '2,\r\n'
            '1,  \r\n'
            '0,00123\r\n'
        ).encode("utf-8")

        results = classify_csv(contents, self.pipeline)

        self.assertEqual(self.pipeline.calls, [["00123", 'Win, "now" please', "00123"]])
        self.assertEqual(
            list(csv.reader(StringIO(export_predictions(results), newline=""))),
            [
                ["message", "prediction"],
                ["00123", "ham"],
                ['Win, "now"\rplease', "spam"],
                ["", ""],
                ["  ", ""],
                ["00123", "ham"],
            ],
        )

    def test_keeps_all_blank_rows_without_calling_pipeline(self):
        results = classify_csv(b'message\n\n""\n" \t"\n', self.pipeline)

        self.assertEqual(
            list(results.itertuples(index=False, name=None)),
            [("", ""), ("", ""), (" \t", "")],
        )
        self.assertEqual(self.pipeline.calls, [])

    def test_rejects_missing_or_duplicate_message_column(self):
        for contents in (b"text\n00123\n", b"message,message\n00123,00123\n"):
            with (
                self.subTest(contents=contents),
                self.assertRaisesRegex(CSVInputError, "message.*column"),
            ):
                classify_csv(contents, self.pipeline)
        self.assertEqual(self.pipeline.calls, [])

    def test_rejects_empty_files(self):
        for contents in (b"", b" \r\n\t", b"\xef\xbb\xbf"):
            with self.subTest(contents=contents), self.assertRaisesRegex(CSVInputError, "empty"):
                classify_csv(contents, self.pipeline)

    def test_rejects_header_only_files(self):
        with self.assertRaisesRegex(CSVInputError, "no message rows"):
            classify_csv(b"message\r\n", self.pipeline)

    def test_rejects_truncated_rows_before_classification(self):
        with self.assertRaisesRegex(CSVInputError, "fields than the header"):
            classify_csv(b"id,message,tag\n1,00123,work\n2,00123\n", self.pipeline)
        self.assertEqual(self.pipeline.calls, [])

    def test_rejects_invalid_encoding_and_malformed_csv(self):
        cases = (
            (b"message\n\xff", "UTF-8"),
            (b'message\n"unclosed', "valid CSV"),
            (b"message\n00123, world\n", "fields than the header"),
        )
        for contents, error in cases:
            with self.subTest(contents=contents), self.assertRaisesRegex(CSVInputError, error):
                classify_csv(contents, self.pipeline)
