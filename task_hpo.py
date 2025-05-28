from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange
import logging
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(
    project_name='Fake News Detection',
    task_name='task_hpo',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Connect parameters
args = {
    'base_train_task_id': '069426c09ce24d5aa36da71ca55147e0',
    'processed_dataset_id': 'f4830f8a2cba413484b470e9df0fad1d',
    'num_trials': 30,
    'time_limit_minutes': 30,
    'run_as_service': False,
    'test_queue': 'pipeline',
    'num_epochs': 20,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
#task.execute_remotely()

# Get the dataset ID from pipeline parameters or fallback
dataset_id = task.get_parameter('General/processed_dataset_id')
if not dataset_id:
    dataset_id = args.get('processed_dataset_id')
    logger.info(f"No dataset_id in General namespace, using from args: {dataset_id}")

if not dataset_id:
    dataset_id = "f4830f8a2cba413484b470e9df0fad1d"
    logger.info(f"Using fallback fixed dataset ID: {dataset_id}")

logger.info(f"Using dataset ID: {dataset_id}")

# Get base task ID
try:
    BASE_TRAIN_TASK_ID = args['base_train_task_id']
    logger.info(f"Using base training task ID: {BASE_TRAIN_TASK_ID}")
except Exception as e:
    logger.error(f"Failed to get base training task ID: {e}")
    raise

# Define HPO logic
hpo_task = HyperParameterOptimizer(
    base_task_id=BASE_TRAIN_TASK_ID,
    hyper_parameters=[
        UniformParameterRange('learning_rate', min_value=0.01, max_value=0.3),
        UniformIntegerParameterRange('max_depth', min_value=3, max_value=10),
        UniformIntegerParameterRange('n_estimators', min_value=50, max_value=200),
        UniformParameterRange('subsample', min_value=0.5, max_value=1.0)
    ],
    objective_metric_title='validation',
    objective_metric_series='accuracy',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=args['num_epochs'],
    pool_period_min=1.0,
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=2,
    parameter_override={
        'processed_dataset_id': dataset_id,
        'General/processed_dataset_id': dataset_id,
        'num_epochs': args['num_epochs'],
        'General/num_epochs': args['num_epochs'],
        'batch_size': args['batch_size'],
        'General/batch_size': args['batch_size'],
        'learning_rate': args['learning_rate'],
        'General/learning_rate': args['learning_rate'],
        'weight_decay': args['weight_decay'],
        'General/weight_decay': args['weight_decay']
    }
)

# Start HPO
logger.info("Starting HPO task...")
hpo_task.start()
logger.info(f"Waiting for {args['time_limit_minutes']} minutes for optimization to complete...")
time.sleep(args['time_limit_minutes'] * 60)

# Retrieve best experiment
try:
    top_exp = hpo_task.get_top_experiments(top_k=1)
    if top_exp:
        best_exp = top_exp[0]
        best_params = best_exp.get_parameters()
        metrics = best_exp.get_last_scalar_metrics()
        accuracy_dict = metrics.get('validation', {}).get('accuracy', {})
        best_accuracy = accuracy_dict[max(accuracy_dict)] if accuracy_dict else None

        logger.info("Best Parameters Found:")
        for k, v in best_params.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"Best validation accuracy: {best_accuracy}")

        result = {
            'parameters': best_params,
            'accuracy': best_accuracy
        }
        with open("best_parameters.json", "w") as f:
            json.dump(result, f, indent=4)

        task.upload_artifact("best_parameters", "best_parameters.json")
        task.set_parameter('best_parameters', best_params)
        task.set_parameter('best_accuracy', best_accuracy)
    else:
        logger.warning("No completed experiments found.")
except Exception as e:
    logger.error(f"Failed to fetch best experiment: {e}")
    raise

# Stop optimizer
hpo_task.stop()
logger.info("HPO Finished.")
print("HPO optimization complete. Goodbye.")
