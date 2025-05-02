from clearml import Task
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

# Initialize ClearML task
task = Task.init(project_name="Fake News Detection", task_name="step4_modeltuning_passive")
Task.force_store_standalone_script(True)

# Load artifacts from Step 2
args = {'dataset_task_id': '67f36284cc4644f5bbe4e77ca1da6933'}
task.connect(args)
dataset_task = Task.get_task(task_id=args['dataset_task_id'])

X_train = dataset_task.artifacts['X_train'].get()
y_train = dataset_task.artifacts['y_train'].get().copy().ravel()
X_test = dataset_task.artifacts['X_test'].get()
y_test = dataset_task.artifacts['y_test'].get().copy().ravel()

# Define hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
    'max_iter': [100, 200, 500, 1000],
    'loss': ['hinge', 'squared_hinge'],
    'fit_intercept': [True, False]
}

# Grid Search
grid = GridSearchCV(PassiveAggressiveClassifier(), param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
print("Tuning PassiveAggressiveClassifier...")
grid.fit(X_train, y_train)

# Best model and score
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)
print("Best Accuracy (CV):", grid.best_score_)
print("Test Accuracy:", accuracy_score(y_test, best_model.predict(X_test)))

# Save and upload model
model_path = "PassiveAggressive_best_model.pkl"
joblib.dump(best_model, model_path)
task.upload_artifact("best_model", artifact_object=model_path)

print("Tuning complete. Model uploaded.")
print(f"Task link: {task.get_output_log_web_page()}")
task.close()
