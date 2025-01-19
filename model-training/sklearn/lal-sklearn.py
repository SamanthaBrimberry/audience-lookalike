# Databricks notebook source
# MAGIC %run ../.././setup/widgets

# COMMAND ----------

# DBTITLE 1,Imports
# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.models import infer_signature
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from databricks.feature_engineering import *
client = mlflow.MlflowClient()

# Spark
from pyspark.sql.functions import *
from pyspark.sql.types import *

# pandas 
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Training sets
# Read the audience_data table once and store it in a DataFrame
audience_data_df = spark.read.table(f'{catalog}.{schema}.lal_features')

# Create the seed_id DataFrame with the specified filters and transformations
# Can also use unsupervised model to identify customer cohorts to target for seed
seed_id = (spark.read.table(f'{catalog}.{schema}.seed_data'))

training_df = (
    audience_data_df
    .join(seed_id, audience_data_df.user_id == seed_id.user_id_seed, "left_outer")
    .withColumn("label", when(col("user_id_seed").isNotNull(), lit(1)).otherwise(lit(0)))
    .withColumn('last_login_date', unix_timestamp(col('last_login_date')))
    .drop('user_id_seed', '_rescued_data')
    .sample(withReplacement=True, fraction=0.30, seed=1)
).toPandas()

unknown_training_df = (
    audience_data_df
    .sample(withReplacement=True, fraction=0.30, seed=2)
    .withColumn('last_login_date', unix_timestamp(col('last_login_date')))
).toPandas()

final_preds_data = (
    audience_data_df
    .sample(withReplacement=True, fraction=0.70, seed=3)
    .withColumn('last_login_date', unix_timestamp(col('last_login_date')))
).toPandas()

# COMMAND ----------

# DBTITLE 1,Set Up MLflow Experiment and Enable Autologging
# set mflow experiment
xp_path = f"/Users/{current_user_email}/Lookalike-Model-Comparison"

mlflow.set_experiment(xp_path)
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

# COMMAND ----------

# DBTITLE 1,Data Preprocessing and Training Split
# training splits
X = training_df.drop(columns='label')
y = training_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# feature prep
categorical_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), num_cols)
    ])

# COMMAND ----------

# DBTITLE 1,Model Selection
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # preprocess, fit, transform
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # eval metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return pipeline, accuracy, precision, recall, f1

# models to try
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": svm.SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# evaluate and tracking expiriments
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline, accuracy, precision, recall, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        signature = infer_signature(X.head(5), pipeline.predict(X))
        
        mlflow.sklearn.log_model(pipeline, "model", signature=signature)

# COMMAND ----------

experiment_id = mlflow.search_experiments(filter_string=f"name LIKE '{xp_path}%'", order_by=["last_update_time DESC"])[0].experiment_id

best_model = mlflow.search_runs(
  experiment_ids=experiment_id,
  order_by=["metrics.f1_score DESC"],
  max_results=1).iloc[0]

# best training run model uri
model_uri = f"runs:/{best_model.run_id}/model"

# COMMAND ----------

loaded_model = mlflow.sklearn.load_model(model_uri)

predictions = loaded_model.predict(unknown_training_df)

unknown_training_df['label'] = predictions

# Combine the unknown training data with the original training data
combined_df = pd.concat(
    [unknown_training_df, training_df],
    ignore_index=True,
    sort=False
).drop_duplicates()

display(combined_df)

# COMMAND ----------

with mlflow.start_run(run_name="best-model"):
    # training splits
    X = combined_df.drop(columns='label')
    y = combined_df['label']

    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # feature prep
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier())
    ])

    model = pipeline.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    signature = infer_signature(X.head(5), model.predict(X))

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature
    )

# COMMAND ----------

# get best run
experiment_id = mlflow.search_experiments(
    filter_string=f"name LIKE '{xp_path}%'",
    order_by=["last_update_time DESC"]
)[0].experiment_id

# isolate run from experiments
latest_run = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["start_time DESC"],
    max_results=1
).iloc[0]

# best run model uri
model_uri = f"runs:/{latest_run.run_id}/model"

# model
loaded_model = mlflow.sklearn.load_model(model_uri)

# register model
model_name = f"{catalog}.{schema}.LookalikeModel"
model_version = mlflow.register_model(model_uri, model_name).version

client.set_registered_model_alias(model_name, "champion", version=model_version)

# # generate preds
# predictions = loaded_model.predict_proba(final_preds_data)[:, 1]  # Extract probabilities for the positive class