# Databricks notebook source
# MAGIC %run ../setup/widgets

# COMMAND ----------

import json
import requests
import os

# COMMAND ----------

base_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
current_user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# get widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

# COMMAND ----------

current_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
current_notebook_dir = os.path.dirname(current_notebook_path)

# target path
target_notebook_name = "dlt-pipeline" 

# path for pipeline
target_notebook_path = os.path.join(current_notebook_dir, target_notebook_name)

# COMMAND ----------

pipeline_config = {
    "development": True,
    "clusters": [
        {
            "label": "default",
            "node_type_id": "m5d.xlarge",
            "autoscale": {
                "min_workers": 1,
                "max_workers": 5,
                "mode": "ENHANCED"
            }
        }
    ],
    "continuous": False,
    "channel": "PREVIEW",
    "photon": True,
    "libraries": [
        {
            "notebook": {
                "path": target_notebook_path
            }
        }
    ],
    "configuration" : {
        "LAL_pipeline.catalog": catalog,
        "LAL_pipeline.schema": schema,
        "LAL_pipeline.volume": volume,
    },
    
    "name": "LAL-features",
    "edition": "ADVANCED",
    "data_sampling": False,
    "allow_duplicate_names": True
}

response = requests.post(
    f'{base_url}/api/2.0/pipelines',
    headers={'Authorization': f'Bearer {token}'},
    data=json.dumps(pipeline_config)
)

pipeline_id = response.json().get('pipeline_id')

# COMMAND ----------

run_config = {}

response = requests.post(
    f'{base_url}/api/2.0/pipelines/{pipeline_id.get('pipeline_id')}/updates',
    headers={'Authorization': f'Bearer {token}'},
    data=json.dumps(run_config)
)

display(response.json())