from clearml import Task
from clearml.automation import PipelineController
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 


def run_pipeline():
    pipe = PipelineController(
        name="Fake News Pipeline",
        project="Fake News Detection",
        version="3.0.0"
    )

     # Set default queue for pipeline control
    pipe.set_default_execution_queue("pipeline")

    # Step 1 - Data Import
    pipe.add_step(
        name="stage_data_import",
        base_task_project="Fake News Detection",
        base_task_name="Pipeline step 1 dataset artifact",
       parameter_override={
            "General/dataset_url": "https://drive.google.com/uc?id=167qj2TvumhjbFPvZHp6CIGZ7aqx5SG_s"
        }
    )

    # Step 2 - Data Preprocessing
    pipe.add_step(
        name="stage_data_processing",
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 2 - Preprocess Dataset",
        parents=["stage_data_import"],
        parameter_override={
            "General/dataset_task_id": "${stage_data_import.id}",
            "General/test_size": 0.3,
            "General/random_state": 1
        }
    )

    # Step 3 - Model Training
    pipe.add_step(
        name="stage_model_training",
        base_task_project="Fake News Detection",
        base_task_name="step3_xgboostmodeltrain",
        parents=["stage_data_processing"],
        parameter_override={
            "General/dataset_task_id": "${stage_data_processing.id}"
        }
    )

    # Step 4: HPO for XGBoost
    pipe.add_step(
        name="step4_hpo",
        parents=["stage_model_training", "stage_data_processing", "stage_data_import"],
        base_task_project="Fake News Detection",
        base_task_name="task_hpo",
        parameter_override={
            "General/processed_dataset_id": "${stage_data_processing.parameters.General/processed_dataset_id}",
            "General/base_train_task_id": "${stage_model_training.id}",
            "General/num_trials": 20,
            "General/time_limit_minutes": 20,
            "General/run_as_service": False,
            "General/dataset_task_id": "${stage_data_import.id}",
        }
    )

    # Step 5: Final model training using best HPO params
    pipe.add_step(
        name="step5_final_model",
        parents=["step4_hpo", "stage_data_processing"],
        base_task_project="Fake News Detection",
        base_task_name="Final Model",
        parameter_override={
            "General/processed_dataset_id": "${stage_data_processing.parameters.General/processed_dataset_id}",
            "General/hpo_task_id": "${step4_hpo.id}"
        }
    )

    pipe.start(queue="pipeline_v2")

    print("Pipeline launched to ClearML")

if __name__ == "__main__":
    run_pipeline()
