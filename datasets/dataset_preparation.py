from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from convokit import Corpus, download, PolitenessStrategies
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset_name):
    """
    Prepares the dataset for training and testing by performing the following steps:
    1. Downloads and loads the corpus based on the given dataset name.
    2. Extracts politeness data from the corpus.
    3. Labels the extracted data.
    4. Filters the labeled data.
    5. Splits the filtered data into training and testing datasets.

    Args:
        dataset_name (str): The name of the dataset to be prepared. 
                            Viable dataset names include:
                            - 'wikipedia'
                            - 'stack-exchange'

    Returns:
        tuple: A tuple containing the training dataframe and the testing dataframe.
    """
    corpus = download_and_load_corpus(dataset_name)
    df = extract_politeness_data(corpus)
    df = label_data(df)
    df_filtered = filter_data(df)
    train_df, test_df = split_data(df_filtered)
    return corpus, train_df, test_df

class PolitenessDataset(Dataset):
    def __init__(self, df):
        texts = df['text']
        labels = df['label']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)
        self.labels = labels.apply(lambda x: 1 if x == 'polite' else 0).tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)

def download_and_load_corpus(dataset_name):
    if dataset_name == 'wikipedia':
        corpus = Corpus(filename=download('wikipedia-politeness-corpus'))
    elif dataset_name == 'stack-exchange':
        corpus = Corpus(filename=download('stack-exchange-politeness-corpus'))
    else:
        raise ValueError("Dataset name must be either 'wikipedia' or 'stack-exchange'")
    return corpus

import pandas as pd

def extract_politeness_data(corpus):
    data = []
    for utt in corpus.iter_utterances():
        text = utt.text
        score = utt.meta['Normalized Score']
        data.append({'text': text, 'score': score})
    return pd.DataFrame(data)

def label_data(df):
    q1 = df['score'].quantile(0.25)
    q3 = df['score'].quantile(0.75)

    def label_score(score):
        if score <= q1:
            return 'impolite'
        elif score >= q3:
            return 'polite'
        else:
            return 'neutral'

    df['label'] = df['score'].apply(label_score)
    return df

def filter_data(df):
    return df[df['label'] != 'neutral'].reset_index(drop=True)

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return train_df, test_df

def extract_features(corpus, df):
    ps = PolitenessStrategies()
    corpus = ps.transform(corpus)
    features = []
    labels = []
    for utt in corpus.iter_utterances():
        if utt.text in df['text'].values:
            features.append(utt.meta['politeness_strategies'])
            labels.append(df[df['text'] == utt.text]['label'].values[0])
    feature_df = pd.DataFrame(features)
    feature_df['label'] = labels
    return feature_df.fillna(0)

def update_corpus(corpus, df):
    texts = set(df['text'].values)
    utterances = [utt for utt in corpus.iter_utterances() if utt.text in texts]
    corpus_filtered = Corpus(utterances=utterances)
    return corpus_filtered
