from clearml import Task
from clearml.automation import PipelineController

def run_pipeline():
    # Initialize the pipeline controller
    pipe = PipelineController(
        name="Fake News Detection Pipeline",        # Pipeline name
        project="Fake News Detection",              # Project name in ClearML
        version="1.0",                              # Pipeline version
        add_pipeline_tags=False
    )

    # Set the default execution queue (must match your ClearML agent's queue)
    pipe.set_default_execution_queue("pipeline")

    # Step 1: Upload raw dataset as a ClearML artifact
    pipe.add_step(
        name="stage_data",                          # Step identifier
        base_task_project="Fake News Detection",    # Project where the task resides
        base_task_name="Pipeline Step 1 - Upload Dataset"  # Name of the existing Task to clone
    )

    # Step 2: Preprocess dataset (clean text, split into train/test, TF-IDF)
    pipe.add_step(
        name="stage_process",                       # Step identifier
        parents=["stage_data"],                     # Depends on the previous step
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 2 - Preprocess Dataset",
        parameter_override={                        # Pass parameters dynamically
            "General/dataset_task_id": "${stage_data.id}",
            "General/test_size": 0.3,
            "General/random_state": 1
        }
    )

    # Step 3: Train the model using processed data
    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],                  # Depends on Step 2
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 3 - Train Model",
        parameter_override={
            "General/dataset_task_id": "${stage_process.id}"
        }
    )

    # Start the pipeline execution using the agent and queue
    pipe.start(queue="pipeline")

    print("🎯 Pipeline successfully launched to ClearML!")

# Allow direct script execution
if __name__ == "__main__":
    run_pipeline()
