import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from train import evaluate, load_messages, remove_duplicates


class TrainingDataTests(unittest.TestCase):
    def test_duplicate_messages_cannot_enter_both_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "sms.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "SMSSpamCollection",
                    "ham\tSee you\tsoon\n"
                    "ham\tsee  you soon\n"
                    "ham\tFree offer\n"
                    "spam\tfree offer\n"
                    "spam\tClaim your prize\n",
                )
            messages = remove_duplicates(load_messages(archive_path))

        self.assertEqual(
            list(messages.itertuples(index=False, name=None)),
            [("ham", "See you soon"), ("spam", "Claim your prize")],
        )

    def test_metrics_measure_spam_and_report_errors_in_order(self):
        actual = ["ham", "ham", "ham", "ham", "spam", "spam", "spam"]
        predicted = ["ham", "ham", "ham", "spam", "ham", "ham", "spam"]

        scores = evaluate(actual, predicted)

        self.assertEqual(scores["accuracy"], 0.5714)
        self.assertEqual(scores["spam_precision"], 0.5)
        self.assertEqual(scores["spam_recall"], 0.3333)
        self.assertEqual(scores["spam_f1"], 0.4)
        self.assertEqual(scores["confusion_matrix"], [[3, 1], [2, 1]])


if __name__ == "__main__":
    unittest.main()
