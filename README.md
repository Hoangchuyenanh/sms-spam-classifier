# SMS Spam Classifier

Classify English SMS messages as spam or normal messages with TF-IDF and logistic regression. This guided learning project compares the model with a majority-class baseline and includes a local Streamlit demo.

[Hướng dẫn từng bước bằng tiếng Việt](GUIDE_VI.md)

## Run locally

Tested on Windows with Python 3.13. From the project folder, run in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Invoke-WebRequest -UseBasicParsing -Uri 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip' -OutFile 'data\sms-spam-collection.zip'
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.headless true
```

Open [the local demo](http://127.0.0.1:8501). Select a sample or enter an English SMS, then select **Classify message**. Press **Ctrl+C** in the terminal to stop the app.

## Model and evaluation

The [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) contains 5,574 labeled messages. After normalizing whitespace, removing case-insensitive duplicates, and dropping conflicting labels, 5,159 messages remained. An 80/20 stratified split with seed 42 produced a held-out test set of 1,032 messages, including 128 spam messages. TF-IDF learns its vocabulary and weights from the training split.

| Model | Accuracy | Spam precision | Spam recall | Spam F1 |
| --- | ---: | ---: | ---: | ---: |
| Majority-class baseline | 87.60% | 0.00% | 0.00% | 0.0000 |
| TF-IDF + logistic regression | 98.16% | 90.98% | 94.53% | 0.9272 |

The baseline classifies every test message as normal, so it misses all 128 spam messages. The trained model catches 121 spam messages, misses 7, and flags 12 normal messages as spam. The [evaluation report](reports/metrics.json) includes both confusion matrices. Its rows are actual labels and columns are predicted labels, ordered normal (`ham`) then `spam`.

This score describes one test split of an older English SMS dataset. Similar messages can still occur in both splits, and results on current or non-English messages may differ.

## Project files

| File | Purpose |
| --- | --- |
| `train.py` | Prepare data, fit both models, and save the trained pipeline and evaluation. |
| `app.py` | Load the trained pipeline and classify one SMS at a time. |
| `reports/metrics.json` | Record the split sizes, scores, and confusion matrices. |
| `tests/test_training.py` | Check duplicate handling and metrics for the spam class. |
| `data/README.md` | Attribute the original dataset and document the input format. |

Dataset by Tiago Almeida and José María Gómez Hidalgo, licensed under CC BY 4.0. The ZIP and trained model are generated or downloaded locally. [Dataset attribution and source](data/README.md)
