# Databricks notebook source
# MAGIC %run .././setup/widgets

# COMMAND ----------

import mlflow
from pyspark.sql.functions import *

# COMMAND ----------

# DBTITLE 1,Prediction Data
audience_data = (
    spark
    .read
    .table(f'{catalog}.{schema}.lal_features')
    .withColumn('last_login_date', unix_timestamp(col('last_login_date')))
).toPandas()

# COMMAND ----------

# DBTITLE 1,Load and Predict with Lookalike Model
# Load the model using an alias
model_uri = f"models:/{catalog}.{schema}.lookalikemodel@Champion"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Predict on the audience data
# Note: can use the predict_proba method since this is logged as an sklearn flavor in mlflow
predictions = loaded_model.predict(audience_data)
audience_data['predictions'] = predictions

# COMMAND ----------

# DBTITLE 1,Filtered Audience Predictions
filtered_data = audience_data.loc[audience_data['predictions'] == 1, ['user_id', 'predictions']]

# COMMAND ----------

filtered_data

# COMMAND ----------

# DBTITLE 1,Save Filtered User IDs to lal_predictions Table
filtered_data = spark.createDataFrame(filtered_data)
(filtered_data.select('user_id').write.mode("overwrite").saveAsTable(f'{catalog}.{schema}.lal_predictions'))