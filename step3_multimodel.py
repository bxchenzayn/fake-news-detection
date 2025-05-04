from clearml import Task
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
import joblib
from pathlib import Path
import os
os.system("pip install pandas")


# Initialize ClearML task
task = Task.init(project_name="Fake News Detection", task_name="Pipeline Step 3 - Train Multiple Models")

# Get parameters (from pipeline)
args = {
    'dataset_task_id': '67f36284cc4644f5bbe4e77ca1da6933' #step2 id
}
task.connect(args)

# Load dataset artifacts from Step 2
dataset_task = Task.get_task(task_id=args['dataset_task_id'])

# Safe loading logic (compatible with both local and pipeline execution)
def safe_load(artifact):
    obj = artifact.get()
    if isinstance(obj, (str, Path)):
        return joblib.load(str(obj))
    return obj

X_train = safe_load(dataset_task.artifacts['X_train'])
X_test = safe_load(dataset_task.artifacts['X_test'])
y_train = safe_load(dataset_task.artifacts['y_train'])
y_test = safe_load(dataset_task.artifacts['y_test'])

# Convert to writable numpy arrays
y_train = np.array(y_train).astype(int).ravel()
y_test = np.array(y_test).astype(int).ravel()

def train_and_log_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    task.get_logger().report_scalar(title=name, series="accuracy", value=acc, iteration=0)

    model_path = f"{name}_model.pkl"
    joblib.dump(model, model_path)
    task.upload_artifact(f"{name}_model", model_path)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=["FAKE", "REAL"], title=f"{name} Confusion Matrix")
    cm_path = f"confusion_matrix_{name}.png"
    plt.savefig(cm_path)
    task.upload_artifact(f"confusion_matrix_{name}", cm_path)
    plt.close()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Train and evaluate each model
train_and_log_model("PassiveAggressive", PassiveAggressiveClassifier(max_iter=1000), X_train, y_train, X_test, y_test)
train_and_log_model("SVM", SVC(kernel='linear'), X_train, y_train, X_test, y_test)
train_and_log_model("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), X_train, y_train, X_test, y_test)

print("All models trained and uploaded to ClearML.")
print(f"Task link: {task.get_output_log_web_page()}")

# Close task
task.close()
