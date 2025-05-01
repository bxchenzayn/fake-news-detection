from clearml import Task
from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # Called before each step runs
    print("Cloning Task id={} with parameters: {}".format(
        a_node.base_task_id, current_param_override))
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # Called after each step finishes
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

    # Step 1: Upload dataset
    pipe.add_step(
        name="stage_data",
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 1 - Upload Dataset",
        repo="https://github.com/bxchenzayn/fake-news-detection.git",
        branch="master",
        working_directory=".",
        entry_point="step1_dataset_artifact.py"
    )

    # Step 2: Preprocess dataset
    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 2 - Preprocess Dataset",
        repo="https://github.com/bxchenzayn/fake-news-detection.git",
        branch="master",
        working_directory=".",
        entry_point="step2_preprocess_dataset.py",
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}",
            "General/test_size": 0.3,
            "General/random_state": 1
        }
    )

    # Step 3: Train model
    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="Fake News Detection",
        base_task_name="Pipeline Step 3 - Train Model",
        repo="https://github.com/bxchenzayn/fake-news-detection.git",
        branch="master",
        working_directory=".",
        entry_point="step3_train_model.py",
        parameter_override={
            "General/dataset_task_id": "${stage_process.id}"
        }
    )

    # Optional: attach pre/post step hooks
    pipe.set_pre_step_callback(pre_execute_callback_example)
    pipe.set_post_step_callback(post_execute_callback_example)

    # Start pipeline
    pipe.start(queue="pipeline")

    print("✅ Pipeline launched")


if __name__ == "__main__":
    run_pipeline()
