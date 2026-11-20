# Dataset

Almeida, T. and Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. [DOI 10.24432/C5CC84](https://doi.org/10.24432/C5CC84). [Dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Save the [original UCI ZIP](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip) in this directory as `sms-spam-collection.zip`. The training script reads the tab-separated `SMSSpamCollection` member directly from the archive.

The script normalizes whitespace, removes case-insensitive duplicates, and drops messages that occur under conflicting labels before splitting the data.
