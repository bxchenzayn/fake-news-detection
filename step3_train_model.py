from clearml import Task
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
import joblib

# Initialize ClearML Task
task = Task.init(project_name="Fake News Detection", task_name="Pipeline Step 3 - Train Model")

# Define arguments and connect
args = {
    'dataset_task_id': '80932355b7324a2fbdebaf61839b330e'  # Step 2 Task ID
}
task.connect(args)

# Load data artifacts from Step 2
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
X_train = dataset_task.artifacts['X_train'].get()
X_test = dataset_task.artifacts['X_test'].get()
y_train = dataset_task.artifacts['y_train'].get()
y_test = dataset_task.artifacts['y_test'].get()

# Convert to writable NumPy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Train the model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy and classification report
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(acc * 100, 2)}%")
print(classification_report(y_test, y_pred))

# Log accuracy
task.get_logger().report_scalar("accuracy", "score", value=acc, iteration=0)

# Save and upload model
joblib.dump(model, 'fake_news_model.pkl')
task.upload_artifact('trained_model', 'fake_news_model.pkl')

# Plot and upload confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
plt.savefig("confusion_matrix.png")
task.upload_artifact("confusion_matrix", "confusion_matrix.png")

print("Model training & evaluation complete.")
print(f"Task link: {task.get_output_log_web_page()}")
print(f"Task ID: {task.id}")

task.close()
