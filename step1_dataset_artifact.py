from clearml import Task
import gdown
import os

# Initialize ClearML Task
task = Task.init(project_name="Fake News Detection", task_name="Pipeline step 1 dataset artifact")
Task.force_store_standalone_script(True)

# Download dataset from Google Drive
file_id = '167qj2TvumhjbFPvZHp6CIGZ7aqx5SG_s'
output_path = 'Fakenews_dataset.csv'
gdown.download(f'https://drive.google.com/uc?id={file_id}', output=output_path, quiet=False)

# Upload dataset to ClearML as an artifact
task.upload_artifact(name='raw_dataset', artifact_object=output_path)

print(f"Dataset uploaded to ClearML from Google Drive as 'raw_dataset'")
print(f"Task link: {task.get_output_log_web_page()}")
print(f"Task ID: {task.id}")

# Mark task as complete
task.close()
