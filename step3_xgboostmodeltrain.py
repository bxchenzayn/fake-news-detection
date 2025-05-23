from clearml import Task
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ClearML Task
task = Task.init(
    project_name="Fake News Detection",
    task_name="step3_xgboostmodeltrain",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

# Connect parameters (for HPO override)
args = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 100,
    'subsample': 1.0
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute remotely
#task.execute_remotely()

# Load preprocessed data from previous task (Step 2)
dataset_task = Task.get_task(task_id="67f36284cc4644f5bbe4e77ca1da6933")
X_train = dataset_task.artifacts['X_train'].get()
y_train = dataset_task.artifacts['y_train'].get().copy().ravel()
X_test = dataset_task.artifacts['X_test'].get()
y_test = dataset_task.artifacts['y_test'].get().copy().ravel()
logger.info(f"Loaded training and test data from task: {dataset_task.id}")

# Train XGBoost model using connected parameters
model = XGBClassifier(
    eval_metric='logloss',
    learning_rate=args['learning_rate'],
    max_depth=args['max_depth'],
    n_estimators=args['n_estimators'],
    subsample=args['subsample']
)
model.fit(X_train, y_train)

# Evaluate
val_preds = model.predict(X_test)
val_accuracy = accuracy_score(y_test, val_preds)
logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

# Report scalar for HPO to use
task.get_logger().report_scalar("validation", "accuracy", val_accuracy, iteration=0)

# Save model
model_path = "XGBoost_model.pkl"
joblib.dump(model, model_path)
task.upload_artifact("model", model_path)
logger.info("Model saved and uploaded.")

# Confusion matrix
cm = confusion_matrix(y_test, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
conf_path = "confusion_matrix.png"
plt.savefig(conf_path)
task.upload_artifact("confusion_matrix", artifact_object=conf_path)
logger.info("Confusion matrix saved and uploaded.")

print("Training and evaluation completed.")
