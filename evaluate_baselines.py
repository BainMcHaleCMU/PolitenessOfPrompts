import openai
from models.bert import train_bert, evaluate_bert
from models.chatgpt import get_and_evaluate_chatgpt_predictions
from datasets.dataset_preparation import prepare_dataset, PolitenessDataset, update_corpus, extract_features
from models.ling import train_svm, evaluate_model

# Load Stack Exchange and Wikipedia data
se_corpus, se_train_df, se_test_df = prepare_dataset('stack-exchange')
wiki_corpus, wiki_train_df, wiki_test_df = prepare_dataset('wikipedia')

# Update datasets for Ling
se_train_corpus = update_corpus(se_corpus, se_train_df)
se_test_corpus = update_corpus(se_corpus, se_test_df)
wiki_train_corpus = update_corpus(wiki_corpus, wiki_train_df)
wiki_test_corpus = update_corpus(wiki_corpus, wiki_test_df)
se_train_features = extract_features(se_train_corpus, se_train_df)
se_test_features = extract_features(se_test_corpus, se_test_df)
wiki_train_features = extract_features(wiki_train_corpus, wiki_train_df)
wiki_test_features = extract_features(wiki_test_corpus, wiki_test_df)

# Evaluate Ling
se_clf = train_svm(se_train_features)
wiki_clf = train_svm(wiki_train_features)
se_ling_accuracy = evaluate_model(se_clf, se_test_features)
wiki_ling_accuracy = evaluate_model(wiki_clf, wiki_test_features)

print(f"Stack Exchange Ling Model Accuracy: {se_ling_accuracy*100:.2f}%")
print(f"Wikipedia Ling Model Accuracy: {wiki_ling_accuracy*100:.2f}%")

# Prepare the datasets for training and testing
se_train_dataset = PolitenessDataset(se_train_df)
se_test_dataset = PolitenessDataset(se_test_df)
wiki_train_dataset = PolitenessDataset(wiki_train_df)
wiki_test_dataset = PolitenessDataset(wiki_test_df)

# Evaluate BERT
print("Training BERT on Stack Exchange data:")
se_trainer = train_bert(se_train_dataset, se_test_dataset)
print("Training BERT on Wikipedia data:")
wiki_trainer = train_bert(wiki_train_dataset, wiki_test_dataset)
print("Evaluating BERT Performance:")
se_bert_accuracy = evaluate_bert(se_trainer, se_test_dataset)
wiki_bert_accuracy = evaluate_bert(wiki_trainer, wiki_test_dataset)
print(f"Stack Exchange BERT Accuracy: {se_bert_accuracy*100:.2f}%")
print(f"Wikipedia BERT Accuracy: {wiki_bert_accuracy*100:.2f}%")

# Run ChatGPT
openai.api_key = 'YOUR_API_KEY'
openai.api_base = 'https://cmu.litellm.ai'
openai.api_type = 'open_ai'  # (Optional) Set the API type if required by the proxy
print("Running ChatGPT on Stack Exchange data:")
se_chatgpt_accuracy  = get_and_evaluate_chatgpt_predictions(se_test_df)
print("Running ChatGPT on Wikipedia data:")
wiki_chatgpt_accuracy = get_and_evaluate_chatgpt_predictions(wiki_test_df)
print(f"Stack Exchange ChatGPT Accuracy: {se_chatgpt_accuracy*100:.2f}%")
print(f"Wikipedia ChatGPT Accuracy: {wiki_chatgpt_accuracy*100:.2f}%")