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

    # Step 1 - Use completed task
    pipe.add_step(
        name="stage_data_import",
        base_task_id="36c9768c38f84f4883aa06a4b9648d2a"
    )

    # Step 2 - Use completed task
    pipe.add_step(
        name="stage_data_processing",
        base_task_id="b6ffb01acf514bff91439f27374d6b06",
        parents=["stage_data_import"]
    )

    # Step 3 - Model Training (Start from here)
    pipe.add_step(
        name="stage_model_multiple_training",
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 3 - Train Multiple Models",
        parents=["stage_data_processing"],
        parameter_override={
            "General/dataset_task_id": "${stage_data_processing.id}"
        }
    )


    

    pipe.start(queue="pipeline_v2")

    print("Pipeline launched to ClearML")


if __name__ == "__main__":
    run_pipeline()
