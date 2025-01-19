# Databricks notebook source
# MAGIC %md
# MAGIC ### Configure Widgets
# MAGIC Before getting started, you can [update the widget values](./setup/widgets.ipynb) as needed.

# COMMAND ----------

# MAGIC %run ./setup/widgets

# COMMAND ----------

# MAGIC %run ./setup/datasetup

# COMMAND ----------

# MAGIC %run ./pipeline/create-pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Training
# MAGIC This project walks your through how to build customer LAL model with two popular modeling frameworks. 
# MAGIC
# MAGIC SparkML is developed specifically for distributed training, leading to better performance when training on large datasets.

# COMMAND ----------

# MAGIC %run ./model-training/sklearn

# COMMAND ----------

# %run ./model-training/sparkml

# COMMAND ----------

# MAGIC %md
# MAGIC ### Activate 
# MAGIC We can make batch inferences using our LAL model we trained and logged. Then we will create a delta-share to share our model and inferences.

# COMMAND ----------

# %run ./activation/batch-inference

# COMMAND ----------

# %run ./activation/delta-share