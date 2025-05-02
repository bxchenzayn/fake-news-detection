from clearml import Task
from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    print(f"Cloning Task id={a_node.base_task_id} with parameters: {current_param_override}")
    return True


def post_execute_callback_example(a_pipeline, a_node):
    print(f"Completed Task id={a_node.executed}")
    return


def run_pipeline():
    # Initialize pipeline
    pipe = PipelineController(
        name="Fake News Detection Pipeline",
        project="Fake News Detection",
        version="2.0.0",
        add_pipeline_tags=False
    )

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
        name="stage_model_multiple_training",
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 3 - Train Multiple Models",
        parents=["stage_data_processing"],
        parameter_override={
            "General/dataset_task_id": "${stage_data_processing.id}"
        }
    )

    # Step 4 - Model Tuning
    pipe.add_step(
        name="stage_model_tuning_passive",
        base_task_project="Fake News Detection",
        base_task_name="step4_modeltuning_passive",
        parents=["stage_data_processing"]
    )

    pipe.add_step(
        name="stage_model_tuning_svm",
        base_task_project="Fake News Detection",
        base_task_name="step4_modeltuning_svm",
        parents=["stage_data_processing"]
    )

    pipe.add_step(
        name="stage_model_tuning_xgboost",
        base_task_project="Fake News Detection",
        base_task_name="step4_modeltuning_xgboost",
        parents=["stage_data_processing"]
    )

    # Step 5 - Model Evaluation
    pipe.add_step(
        name="stage_model_evaluation",
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 5 - Model Evaluation",
        parents=[
            "stage_model_tuning_passive",
            "stage_model_tuning_svm",
            "stage_model_tuning_xgboost"
        ]
    )

    pipe.start(queue="pipeline_start")

    print("Pipeline launched to ClearML")


if __name__ == "__main__":
    run_pipeline()
