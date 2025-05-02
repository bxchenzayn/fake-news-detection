from clearml import Task
import joblib
from sklearn.metrics import classification_report, accuracy_score
import os

# Initialize ClearML Task
task = Task.init(project_name="Fake News Detection", task_name="Pipeline Step 5 - Model Evaluation")

# Download test data artifacts from previous task
args = {
    'dataset_task_id': '67f36284cc4644f5bbe4e77ca1da6933'
}
task.connect(args)

# Load artifacts
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
X_test = dataset_task.artifacts['X_test'].get()
y_test = dataset_task.artifacts['y_test'].get()

# Convert y_test to numeric and 1D
import numpy as np
y_test = np.array(y_test).astype(int).ravel()

# Define model paths
model_paths = {
    "PassiveAggressive": "PassiveAggressive_best_model.pkl",
    "LinearSVC": "LinearSVC_best_model.pkl",
    "XGBoost": "XGBoost_best_model.pkl"
}

# Evaluate each model
results = {}
for name, path in model_paths.items():
    if not os.path.exists(path):
        print(f"{name} model file not found: {path}")
        continue
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    results[name] = {
        "accuracy": acc,
        "classification_report": report
    }
    print(f"\n {name} Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(report)

# Upload results to ClearML
task.upload_artifact("model_evaluation_results", results)

# Close the task
task.close()
