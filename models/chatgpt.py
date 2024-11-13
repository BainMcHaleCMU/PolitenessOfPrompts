import openai
import time

def get_chatgpt_prediction(text):
    prompt = f"Do a binary classification on the politeness for the given text. If it's polite then output 1. Otherwise output 0 for impolite.\n\n{text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use a model supported by the proxy
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1,
        temperature=0,
    )

    prediction = response['choices'][0]['message']['content'].strip()
    return prediction

def get_chatgpt_predictions(df):
    predictions = []
    for _, row in df.iterrows():
        text = row['text']
        prediction = get_chatgpt_prediction(text)
        predictions.append(prediction)
        time.sleep(1)  # Adjust based on rate limits
    return predictions

def process_predictions(predictions):
    processed = []
    for pred in predictions:
        if '1' in pred:
            processed.append('polite')
        elif '0' in pred:
            processed.append('impolite')
        else:
            processed.append('unknown')
    return processed

def evaluate_chatgpt(df, predictions):
    df = df.copy()
    df['prediction'] = predictions
    df = df[df['prediction'] != 'unknown']
    accuracy = (df['label'] == df['prediction']).mean()
    return accuracy

def get_and_evaluate_chatgpt_predictions(df):
    raw_predictions = get_chatgpt_predictions(df)
    processed_predictions = process_predictions(raw_predictions)
    accuracy = evaluate_chatgpt(df, processed_predictions)
    return accuracy