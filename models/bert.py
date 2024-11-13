from transformers import BertForSequenceClassification, Trainer, TrainingArguments

def train_bert(train_dataset, eval_dataset):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        save_total_limit=1,
        # The fix: Set save_strategy to "epoch" to match evaluation_strategy
        save_strategy="epoch",
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        acc = (preds == labels).mean()
        return {'accuracy': acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer

def evaluate_bert(trainer, test_dataset):
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    accuracy = metrics['eval_accuracy']
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy