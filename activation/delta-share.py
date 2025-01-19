# Databricks notebook source
# MAGIC %run .././setup/widgets

# COMMAND ----------

# DBTITLE 1,create lookalike model share in delta
# Create a Delta Share for lookalike model, lookalike features, and prediction table
spark.sql("""
CREATE SHARE lookalike_model_share
COMMENT 'Share for lookalike model, lookalike features, and prediction table'
""")

# COMMAND ----------

# DBTITLE 1,add lal features to delta share
# Add materialized view to the share...Coming Soon!
# spark.sql(f"""
# ALTER SHARE lookalike_model_share
# ADD MATERIALIZED_VIEW {catalog}.{schema}.lal_features
# """)

# COMMAND ----------

# DBTITLE 1,add lal predictions to delta share
spark.sql(f"""
ALTER SHARE lookalike_model_share
ADD TABLE {catalog}.{schema}.lal_predictions WITH HISTORY
""")

# COMMAND ----------

# DBTITLE 1,add lal model to lookalike model share
spark.sql(f"""
ALTER SHARE lookalike_model_share
ADD MODEL {catalog}.{schema}.lookalikemodel
COMMENT "audience classification lal model"
AS {schema}.lal_model
""")