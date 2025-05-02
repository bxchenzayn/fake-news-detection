from clearml import Task
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

# Initialize ClearML Task
task = Task.init(project_name="Fake News Detection", task_name="step4_modeltuning_svm")
Task.force_store_standalone_script(True)

# Load artifacts from Step 2 task
dataset_task = Task.get_task(task_id="67f36284cc4644f5bbe4e77ca1da6933")
X_train = dataset_task.artifacts['X_train'].get()
X_test = dataset_task.artifacts['X_test'].get()
y_train = dataset_task.artifacts['y_train'].get().copy().ravel()
y_test = dataset_task.artifacts['y_test'].get().copy().ravel()

# Define parameter grid for LinearSVC
param_grid = {
    'C': [0.1, 1, 5, 10],
    'max_iter': [1000, 2000],
    'dual': [True, False]
}

# Perform grid search using 3-fold cross-validation
grid = GridSearchCV(
    estimator=LinearSVC(),
    param_grid=param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("Tuning LinearSVC...")
grid.fit(X_train, y_train)

# Evaluate best model
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)
print("Best Accuracy (CV):", grid.best_score_)
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# Save and upload model artifact
model_path = "LinearSVC_best_model.pkl"
joblib.dump(best_model, model_path)
task.upload_artifact("best_model", artifact_object=model_path)

print("Tuning complete.")
print(f"Task link: {task.get_output_log_web_page()}")
print(f"Task ID: {task.id}")
task.close()
