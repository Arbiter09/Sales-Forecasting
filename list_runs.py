import mlflow

# Get all experiments
experiments = mlflow.list_experiments()
for experiment in experiments:
    print(f"ID: {experiment.experiment_id}, Name: {experiment.name}")
