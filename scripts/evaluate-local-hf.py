from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import tarfile
import os

if len(os.listdir('model')) == 0:
    with tarfile.open('model.tar.gz') as tar:
        tar.extractall(path="model")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


model = AutoModelForSequenceClassification.from_pretrained('model')
test_dataset = load_from_disk('test_data')
test_dataset = test_dataset.select(range(100))

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

eval_result = trainer.evaluate(eval_dataset=test_dataset)
print(eval_result)
