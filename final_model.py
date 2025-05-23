from clearml import Task, Dataset
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ClearML Task
task = Task.init(
    project_name="Fake News Detection",
    task_name="Final Model",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

# Connect parameters (if passed from pipeline)
args = {
    'processed_dataset_task_id': '67f36284cc4644f5bbe4e77ca1da6933',
    'hpo_task_id': '48238df0c30d4eada007c956befe5407',  # Optional HPO task ID for loading best parameters
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")


# Load preprocessed data from Step 2
dataset_task = Task.get_task(task_id=args['processed_dataset_task_id'])
X_train = dataset_task.artifacts['X_train'].get()
y_train = dataset_task.artifacts['y_train'].get().to_numpy()
X_test = dataset_task.artifacts['X_test'].get()
y_test = dataset_task.artifacts['y_test'].get().to_numpy()

# Default parameters
best_params = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 100,
    'subsample': 1.0
}

# Load best parameters from HPO task if provided
if args.get('hpo_task_id'):
    try:
        hpo_task = Task.get_task(task_id=args['hpo_task_id'])
        artifact_path = hpo_task.artifacts['best_parameters'].get_local_copy()
        with open(artifact_path, 'r') as f:
            best_data = json.load(f)
            best_params.update({
                'learning_rate': float(best_data['parameters'].get('General/learning_rate', best_params['learning_rate'])),
                'max_depth': int(best_data['parameters'].get('General/max_depth', best_params['max_depth'])),
                'n_estimators': int(best_data['parameters'].get('General/n_estimators', best_params['n_estimators'])),
                'subsample': float(best_data['parameters'].get('General/subsample', best_params['subsample']))
            })
        logger.info(f"Loaded best parameters from HPO: {best_params}")
    except Exception as e:
        logger.warning(f"Could not load best parameters from HPO: {e}")

# Train final XGBoost model
model = XGBClassifier(
    eval_metric='logloss',
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    subsample=best_params['subsample']
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
logger.info(f"Final model accuracy: {acc:.4f}")
task.get_logger().report_scalar("FinalModel", "accuracy", acc, iteration=0)

# Save model
model_path = "final_xgboost_model.pkl"
joblib.dump(model, model_path)
task.upload_artifact("final_model", model_path)

# Confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("final_confusion_matrix.png")
task.upload_artifact("confusion_matrix", "final_confusion_matrix.png")

logger.info("Final training complete.")
print("Final XGBoost model trained and uploaded.")