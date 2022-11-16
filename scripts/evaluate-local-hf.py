from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import tarfile
import os

if len(os.listdir('model')) == 0:
    with tarfile.open('model.tar.gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path="model")


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
