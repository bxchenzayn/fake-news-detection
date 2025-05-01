from clearml import Task
from clearml.automation import PipelineController

def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print("Cloning Task id={} with parameters: {}".format(
        a_node.base_task_id, current_param_override
    ))
    return True

def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    return

def run_pipeline():
    pipe = PipelineController(
        name="Fake News Detection Pipeline",
        project="Fake News Detection",
        version="1.0.0",
        add_pipeline_tags=False
    )

    pipe.set_default_execution_queue("pipeline")

    pipe.add_step(
        name="stage_data",
        base_task_project="Fake News Detection",
        base_task_name="Pipeline step 1 dataset artifact"
    )

    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 2 - Preprocess Dataset",
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}",
            "General/test_size": 0.3,
            "General/random_state": 1
        }
    )

    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 3 - Train Model",
        parameter_override={
            "General/dataset_task_id": "${stage_process.id}"
        }
    )

    pipe.start(queue="pipeline")
    print("done")

if __name__ == "__main__":
    run_pipeline()
