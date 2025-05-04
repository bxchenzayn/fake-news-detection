from clearml import Task
import joblib
from sklearn.metrics import classification_report, accuracy_score
import os
import numpy as np

# Initialize ClearML Task
task = Task.init(project_name="Fake News Detection", task_name="Pipeline Step 5 - Model Evaluation")

# Download test data artifacts
args = {
    'dataset_task_id': '67f36284cc4644f5bbe4e77ca1da6933',
    'passive_task_id': '75bd29c55af1474d83b346ac73679976',
    'svm_task_id': '93df74834d714b71a22d32dd4014695a',
    'xgboost_task_id': '800950648fe144a090ecddddb81a38cc'
}
task.connect(args)

# Load dataset
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
X_test = dataset_task.artifacts['X_test'].get()
y_test = dataset_task.artifacts['y_test'].get()


y_test = np.array(y_test).astype(int).ravel()

# Define model task sources
model_tasks = {
    "PassiveAggressive": args['passive_task_id'],
    "LinearSVC": args['svm_task_id'],
    "XGBoost": args['xgboost_task_id']
}

# Evaluate each model
results = {}
for name, model_task_id in model_tasks.items():
    print(f"Loading {name} from task {model_task_id}")
    model_task = Task.get_task(task_id=model_task_id)
    model_path = model_task.artifacts['best_model'].get()
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    results[name] = {
        "accuracy": acc,
        "classification_report": report
    }
    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(report)

# Upload evaluation results
task.upload_artifact("model_evaluation_results", results)


task.close()
