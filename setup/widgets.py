# Databricks notebook source
dbutils.widgets.text('catalog', "sb")
dbutils.widgets.text('schema', "lal")
dbutils.widgets.text('volume', "ads")

catalog = dbutils.widgets.get('catalog')
schema = dbutils.widgets.get('schema')
volume = dbutils.widgets.get('volume')

folder = f"/Volumes/{catalog}/{schema}/{volume}"


# COMMAND ----------

# MAGIC %sql
# MAGIC create catalog if not exists ${catalog};
# MAGIC
# MAGIC use catalog ${catalog};
# MAGIC
# MAGIC create schema if not exists ${schema};
# MAGIC
# MAGIC CREATE VOLUME if not exists ${catalog}.${schema}.${volume};