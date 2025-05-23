from clearml import Task
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import atexit
import os

# Initialize ClearML Task
task = Task.init(project_name="Fake News Detection", task_name="step3_xgboostmodeltrain")

# Define and connect hyperparameters for HPO to override
params = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 100,
    'subsample': 1.0
}
params = task.connect(params)

# Load preprocessed data from previous task (Step 2)
dataset_task = Task.get_task(task_id="67f36284cc4644f5bbe4e77ca1da6933")
X_train = dataset_task.artifacts['X_train'].get()
y_train = dataset_task.artifacts['y_train'].get().copy().ravel()
X_test = dataset_task.artifacts['X_test'].get()
y_test = dataset_task.artifacts['y_test'].get().copy().ravel()

# Train XGBoost model using connected parameters
model = XGBClassifier(
    eval_metric='logloss',
    learning_rate=params['learning_rate'],
    max_depth=params['max_depth'],
    n_estimators=params['n_estimators'],
    subsample=params['subsample']
)
model.fit(X_train, y_train)

# Evaluate validation accuracy
val_accuracy = accuracy_score(y_test, model.predict(X_test))
print("Validation Accuracy:", val_accuracy)

# Report scalar metric to ClearML for HPO to consume
task.get_logger().report_scalar(
    title="validation",
    series="accuracy",
    value=val_accuracy,
    iteration=1
)

# Save and upload the trained model
model_path = "XGBoost_default_model.pkl"
joblib.dump(model, model_path)
task.upload_artifact("default_model", artifact_object=model_path)

# Clean up saved file on exit
@atexit.register
def cleanup():
    try:
        os.remove(model_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

# Close the ClearML task
task.close()
print(f"Training completed. Task ID: {task.id}")
