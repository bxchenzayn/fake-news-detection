from clearml import Task
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Initialize ClearML task
task = Task.init(project_name="Fake News Detection", task_name="step4_modeltuning_xgboost")
Task.force_store_standalone_script(True)

# Load artifacts from previous task
dataset_task = Task.get_task(task_id="67f36284cc4644f5bbe4e77ca1da6933")  
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

# Extract the best model and results
best_model = grid.best_estimator_
best_params = grid.best_params_
best_score = grid.best_score_
test_score = best_model.score(X_test, y_test)

# Print results
print("Best Params:", best_params)
print("Best Accuracy (CV):", best_score)
print("Test Accuracy:", test_score)

# Save and upload the best model
joblib.dump(best_model, "XGBoost_best_model.pkl")
task.upload_artifact("best_model", artifact_object="XGBoost_best_model.pkl")

# Close the task
task.close()
