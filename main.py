import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import re
import nltk


# download nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def preprocess_text(text):
    """
        clean data
        remove html tag, special characters, convert to lowercase,  remove stop words.
    """
    if pd.isna(text):
        return ""
    # remove html
    text = re.sub(r'<[^>]+>', ' ', text)
    # remove url
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # only save word, figure
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\u0400-\u04FF]+', ' ', text)
    # tolower
    text = text.lower().strip()

    text = ' '.join(text.split())
    return text


if __name__ == '__main__':
    # set seed
    seed = 2026
    set_seed(seed)
    # load data
    train_df = pd.read_csv('csv/train.csv')

    print(f"train data shape: {train_df.shape}")
    # clean code
    train_df['cleaned_text'] = train_df['TEXT'].apply(preprocess_text)
    train_df = train_df[train_df['cleaned_text'].str.len() > 0]


    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=False
    )

    # encode label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df['LABEL'])

    # model
    model = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced'))
    ])

    # train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['cleaned_text'], y, test_size=0.2, random_state=seed, stratify=y
    )

    # fit model
    model.fit(X_train, y_train)

    # from sklearn.metrics import classification_report, accuracy_score
    # y_pred = model.predict(X_val)
    # print(f"ACC: {accuracy_score(y_val, y_pred):.2f}")
    # print(classification_report(y_val, y_pred, target_names=['Negative', 'Positive', 'Not Review']))

    # predict
    test_df = pd.read_csv('csv/test.csv')
    print(f"test  data shape: {test_df.shape}")

    test_df['cleaned_text'] = test_df['TEXT'].apply(preprocess_text)
    final_predictions = model.predict(test_df['cleaned_text'])
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'LABEL': final_predictions
    })
    submission = test_df[['ID']].copy()
    submission['LABEL'] = final_predictions
    submission.to_csv('csv/predict.csv', index=False)
    print(submission.shape)

