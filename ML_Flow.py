import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient  # Import MlflowClient to manage experiments
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Load your dataset
dataset_path = r'C:\Users\shahj\OneDrive\Desktop\Projects\Final-Retail-Sales-Forecasting-main\Cleaned_Store_data2.csv'

# Validate dataset
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

df = pd.read_csv(dataset_path)
if df.empty:
    raise ValueError("The dataset is empty. Please provide a valid dataset.")

# Ensure all required columns are present
required_columns = ['Weekly_Sales', 'Size', 'Type', 'Date', 'weekly_sales', 'Markdown']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in the dataset: {missing_columns}")

data = df.copy()

def inv_trans(x):
    if x == 0:
        return x
    else:
        return 1 / x

# Apply inverse transformation
data['Markdown'] = data['Markdown'].apply(inv_trans)

# Prepare the data
x = data.drop(['Weekly_Sales', 'Size', 'Type', 'Date', 'weekly_sales'], axis=1)
y = data['Weekly_Sales']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)

# Set MLflow tracking URI to a dedicated directory
mlflow_tracking_uri = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"MLflow tracking URI is set to: {mlflow.get_tracking_uri()}")

# Define the directory for local models
local_models_dir = r"C:\Users\shahj\OneDrive\Desktop\Projects\Final-Retail-Sales-Forecasting-main\local_models"
os.makedirs(local_models_dir, exist_ok=True)

# Create or get the MLflow experiment
experiment_name = "Retail Sales Forecasting"
client = MlflowClient()

# Check if the experiment already exists
experiment = client.get_experiment_by_name(experiment_name)
if experiment is not None:
    experiment_id = experiment.experiment_id
    print(f"Experiment '{experiment_name}' already exists with ID {experiment_id}")
else:
    # Create the experiment
    experiment_id = client.create_experiment(experiment_name)
    print(f"Experiment '{experiment_name}' created with ID {experiment_id}")

# Define the Random Forest Model
def randomforest():
    n_estimators_val = [50, 100, 200, 300, 350, 400]
    Max_depth = [10, 10, 10, 25, 25, 25]
    Min_samples_split = [5, 5, 5, 15, 15, 15]
    Min_samples_leaf = [7, 7, 7, 13, 13, 13]

    for i, j, k, l in zip(Max_depth, Min_samples_split, Min_samples_leaf, n_estimators_val):
        # Initialize and train the model
        model_rf = RandomForestRegressor(
            criterion='squared_error',
            n_estimators=l,
            max_depth=i,
            min_samples_split=j,
            min_samples_leaf=k,
            random_state=30
        )
        model_rf.fit(x_train, y_train)

        y_train_pred = model_rf.predict(x_train)
        y_test_pred = model_rf.predict(x_test)

        # Compute metrics
        train_MAE = mean_absolute_error(y_train, y_train_pred)
        test_MAE = mean_absolute_error(y_test, y_test_pred)

        train_MAPE = mean_absolute_percentage_error(y_train, y_train_pred)
        test_MAPE = mean_absolute_percentage_error(y_test, y_test_pred)

        # Prepare input example and signature
        input_example = x_train.head(1)
        signature = infer_signature(x_train, y_train)

        # Start an MLflow run with a custom name and specify the experiment ID
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"RF_n_estimators_{l}") as run:
            run_id = run.info.run_id
            print(f"Run ID: {run_id}")
            print(f"Experiment ID: {run.info.experiment_id}")

            # Set tags for better identification
            mlflow.set_tag("model_type", "RandomForestRegressor")
            mlflow.set_tag("run_id", run_id)

            # Log model parameters
            mlflow.log_param('n_estimators', l)
            mlflow.log_param('max_depth', i)
            mlflow.log_param('min_samples_split', j)
            mlflow.log_param('min_samples_leaf', k)

            # Log evaluation metrics
            mlflow.log_metric('train_MAE', train_MAE)
            mlflow.log_metric('test_MAE', test_MAE)
            mlflow.log_metric('train_MAPE', train_MAPE)
            mlflow.log_metric('test_MAPE', test_MAPE)

            # Log the trained model into MLflow artifacts with signature
            try:
                print("Logging model to MLflow artifacts...")
                mlflow.sklearn.log_model(
                    sk_model=model_rf,
                    artifact_path='model',
                    input_example=input_example,
                    signature=signature
                )
                print(f"Run ID: {run_id} - Model logged to artifacts")
            except Exception as e:
                print(f"Error logging model to MLflow artifacts: {e}")

            # Save the model locally using an absolute path
            local_model_path = os.path.join(local_models_dir, f'local_model_{run_id}')
            try:
                print(f"Saving model to local directory '{local_model_path}'...")
                mlflow.sklearn.save_model(
                    sk_model=model_rf,
                    path=local_model_path,
                    input_example=input_example,
                    signature=signature
                )
                print("Model saved locally.")
            except Exception as e:
                print(f"Error saving model locally: {e}")

            # Test loading the model from MLflow to verify it was saved correctly
            try:
                model_uri = f"runs:/{run_id}/model"
                print(f"Loading model from: {model_uri}")
                loaded_model = mlflow.sklearn.load_model(model_uri)
                sample_predictions = loaded_model.predict(x_test.head())
                print("Sample predictions from loaded model:", sample_predictions)
            except Exception as e:
                print(f"Error loading model from MLflow artifacts: {e}")

# Run the random forest training and logging
print(f"MLflow tracking URI is set to: {mlflow.get_tracking_uri()}")
randomforest()