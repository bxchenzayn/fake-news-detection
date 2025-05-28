from clearml import Task
import gdown

# Initialize ClearML Task and receive dynamic parameters
task = Task.init(project_name="Fake News Detection v1", task_name="Pipeline step 1 dataset artifact")
Task.force_store_standalone_script(True)

# Receive dataset_url from pipeline
args = {
    'dataset_url': 'https://drive.google.com/uc?id=167qj2TvumhjbFPvZHp6CIGZ7aqx5SG_s'
}
task.connect(args)

# Download dataset using gdown
output_path = 'Fakenews_dataset.csv'
gdown.download(args['dataset_url'], output=output_path, quiet=False)

# Upload to ClearML
task.upload_artifact(name='raw_dataset', artifact_object=output_path)

print("Dataset uploaded as artifact.")
print(f"Task link: {task.get_output_log_web_page()}")
print(f"Task ID: {task.id}")

task.close()
