import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
NORMAL_MESSAGE = "Hey, are we still meeting for lunch tomorrow?"
SPAM_MESSAGE = "Congratulations! You have won a free cash prize. Reply WIN now to claim."


class AppClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = cls.enterClassContext(TemporaryDirectory())
        model_path = Path(directory) / "sms-pipeline.joblib"
        pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
        pipeline.fit([NORMAL_MESSAGE, SPAM_MESSAGE], ["ham", "spam"])
        joblib.dump(pipeline, model_path)
        cls.enterClassContext(patch("train.MODEL_PATH", model_path))

    def test_batch_results_stay_until_the_upload_changes(self):
        app = AppTest.from_file(str(APP_PATH)).run()
        self.assertEqual(len(app.file_uploader), 1, "The batch CSV uploader should be available.")
        contents = f'message\n"{NORMAL_MESSAGE}"\n""\n{SPAM_MESSAGE}\n'.encode("utf-8")
        app.file_uploader[0].set_value(("messages.csv", contents, "text/csv")).run()
        app.button(key="classify_csv").click().run()

        expected = [(NORMAL_MESSAGE, "ham"), ("", ""), (SPAM_MESSAGE, "spam")]
        self.assertFalse(app.exception)
        self.assertEqual(list(app.dataframe[0].value.itertuples(index=False, name=None)), expected)
        self.assertEqual(len(app.download_button), 1)
        self.assertTrue(any("blank message" in warning.value for warning in app.warning))

        app.run()
        self.assertEqual(len(app.download_button), 1)
        app.file_uploader[0].set_value(("messages.csv", b"message\nnew text\n", "text/csv")).run()
        self.assertFalse(app.dataframe)
        self.assertFalse(app.download_button)

    def test_invalid_uploads_show_errors_without_results(self):
        app = AppTest.from_file(str(APP_PATH)).run()
        self.assertEqual(len(app.file_uploader), 1, "The batch CSV uploader should be available.")
        for contents, message in ((b"", "empty"), (b"text\nhello\n", "column")):
            with self.subTest(contents=contents):
                app.file_uploader[0].set_value(("messages.csv", contents, "text/csv")).run()
                app.button(key="classify_csv").click().run()
                self.assertFalse(app.exception)
                self.assertIn(message, app.error[0].value)
                self.assertFalse(app.download_button)

    def test_single_message_examples_and_classification_still_work(self):
        for example_button, message, status, expected in (
            ("Try a normal SMS", NORMAL_MESSAGE, "success", "Prediction: normal message"),
            ("Try a spam SMS", SPAM_MESSAGE, "warning", "Prediction: spam"),
        ):
            with self.subTest(message=message):
                app = AppTest.from_file(str(APP_PATH)).run()
                next(button for button in app.button if button.label == example_button).click().run()
                self.assertEqual(app.text_area[0].value, message)
                next(button for button in app.button if button.label == "Classify message").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(getattr(app, status)[0].value, expected)
