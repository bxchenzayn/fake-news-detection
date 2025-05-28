from clearml import Task, Logger
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
import atexit

# Initialize ClearML task
task = Task.init(project_name="Fake News Detection", task_name="step4_modeltuning_xgboost")

# Load artifacts from previous task
dataset_task = Task.get_task(task_id="f4830f8a2cba413484b470e9df0fad1d")  
X_train = dataset_task.artifacts['X_train'].get()
y_train = dataset_task.artifacts['y_train'].get().copy().ravel()
X_test = dataset_task.artifacts['X_test'].get()
y_test = dataset_task.artifacts['y_test'].get().copy().ravel()

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

# Initialize XGBoost classifier
xgb_model = XGBClassifier(eval_metric='logloss')

# Perform grid search with 3-fold cross validation
grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

# Train the model
grid.fit(X_train, y_train)

# Best model and score
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)
print("Best Accuracy (CV):", grid.best_score_)

# Compute and report validation accuracy
val_accuracy = accuracy_score(y_test, best_model.predict(X_test))
print("Test Accuracy:", val_accuracy)

# Report scalar to ClearML for HPO tracking
Logger.current_logger().report_scalar(
    title="validation",
    series="accuracy",
    value=val_accuracy,
    iteration=1
)

# Save and upload the best model
model_path = "XGBoost_best_model.pkl"
joblib.dump(best_model, model_path)
task.upload_artifact("best_model", artifact_object=model_path)

# Cleanup saved file at exit
@atexit.register
def cleanup():
    try:
        os.remove(model_path)
        print(f"Removed temporary file: {model_path}")
    except Exception as e:
        print(f"Could not remove model file: {e}")

# Close the task
print("Tuning complete. Model uploaded.")
print(f"Task link: {task.get_output_log_web_page()}")
task.close()
