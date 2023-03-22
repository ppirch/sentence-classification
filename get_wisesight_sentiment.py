import requests
import pandas as pd

TRAIN_TEXT_URL = "https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/kaggle-competition/train.txt"
TRAIN_LABEL_URL = "https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/kaggle-competition/train_label.txt"
TEST_TEXT_URL = "https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/kaggle-competition/test.txt"
TEST_LABEL_URL = "https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/kaggle-competition/test_label.txt"

if __name__ == "__main__":
    train_text = requests.get(TRAIN_TEXT_URL).text
    train_label = requests.get(TRAIN_LABEL_URL).text
    test_text = requests.get(TEST_TEXT_URL).text
    test_label = requests.get(TEST_LABEL_URL).text

    train_df = pd.DataFrame(
        {"text": train_text.split("\n")[:-1], "label": train_label.split("\n")[:-1]}
    )

    test_df = pd.DataFrame(
        {"text": test_text.split("\n")[:-1], "label": test_label.split("\n")[:-1]}
    )

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
