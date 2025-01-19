# Databricks notebook source
# MAGIC %run ./widgets

# COMMAND ----------

import json
import requests

# COMMAND ----------

base_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
current_user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

cluster_config = {
    "cluster_name": "lal-cluster",
    "spark_version": "16.1.x-scala2.12",
    "spark_conf": {
        "spark.kryoserializer.buffer.max": "1024m"
    },
    "aws_attributes": {
        "first_on_demand": 3,
        "availability": "SPOT_WITH_FALLBACK",
        "zone_id": "auto",
        "spot_bid_price_percent": 100,
        "ebs_volume_type": "GENERAL_PURPOSE_SSD",
        "ebs_volume_count": 3,
        "ebs_volume_size": 100
    },
    "node_type_id": "c7i.4xlarge",
    "driver_node_type_id": "c7i.4xlarge",
    "autotermination_minutes": 20,
    "enable_elastic_disk": false,
    "single_user_name": f"{current_user_email}",
    "enable_local_disk_encryption": false,
    "data_security_mode": "DATA_SECURITY_MODE_DEDICATED",
    "runtime_engine": "PHOTON",
    "kind": "CLASSIC_PREVIEW",
    "use_ml_runtime": true,
    "is_single_node": false,
    "autoscale": {
        "min_workers": 2,
        "max_workers": 8
    },
    "apply_policy_default_values": false
}

response = requests.post(
    f'{base_url}/api/2.1/clusters/create',
    headers={'Authorization': f'Bearer {token}'},
    data=json.dumps(config)
)

display(response.json())