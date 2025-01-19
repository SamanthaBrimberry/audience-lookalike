# Databricks notebook source
import dlt
from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark.sql.types import *
import mlflow

# COMMAND ----------

@dlt.table()

def audience_data():
  catalog = spark.conf.get("LAL_pipeline.catalog")
  schema = spark.conf.get("LAL_pipeline.schema")
  volume = spark.conf.get("LAL_pipeline.volume")

  return (
    spark.readStream.format("cloudFiles")
    .option('cloudFiles.format', 'json')
    .load(f'/Volumes/{catalog}/{schema}/{volume}/audience_df')
  )

# COMMAND ----------

@dlt.table(comment = 'raw seed data')
          
def seed_data():
  catalog = spark.conf.get("LAL_pipeline.catalog")
  schema = spark.conf.get("LAL_pipeline.schema")
  volume = spark.conf.get("LAL_pipeline.volume")
  
  return(
    spark.readStream.format('cloudFiles')
    .option('cloudFiles.format', 'json')
    .option("cloudFiles.inferColumnTypes", "true")
    .load(f'/Volumes/{catalog}/{schema}/{volume}/seed_df'))

# COMMAND ----------

@dlt.table(comment = 'ad impression data')
          
def ad_impressions():
  catalog = spark.conf.get("LAL_pipeline.catalog")
  schema = spark.conf.get("LAL_pipeline.schema")
  volume = spark.conf.get("LAL_pipeline.volume")

  return(
    spark.readStream.format('cloudFiles')
    .option('cloudFiles.format', 'json')
    .option("cloudFiles.inferColumnTypes", "true")
    .load(f'/Volumes/{catalog}/{schema}/{volume}/ad_impressions')
  )

# COMMAND ----------

def get_rules(tag):
  """
    loads data quality rules from a table
    :param tag: tag to match
    :return: dictionary of rules that matched the tag
  """
  rules = {}
  df = spark.read.table("sb.lal.quality_rules")
  for row in df.filter(col("tag") == tag).collect():
    rules[row['name']] = row['constraint']
  return rules

@dlt.table(name='audience_data_silver',
           schema = """
          ad_interaction_rate float,
          age_group string,
          avg_daily_watch_time double,
          binge_watching string,
          content_sharing string,
          education_level string,
          email string,
          ethnicity string,
          favorite_genre string,
          gender string,
          home_ownership string,
          household_income string,
          language_preference string,
          last_login_date date,
          location string,
          marital_status string,
          name string,
          num_children int,
          occupation string,
          preferred_viewing_time string,
          primary_device string,
          subscription_length double,
          subscription_type string,
          user_id string not null primary key""")

@dlt.expect_all(get_rules('audience_team'))

def audience_data_silver():
  return (dlt.read("audience_data")
          .withColumn("subscription_length", col("subscription_length").cast("double"))
          .withColumn('num_children', col('num_children').cast('int'))
          .withColumn('last_login_date', to_date('last_login_date', 'yyyy-MM-dd'))
          .withColumn("avg_daily_watch_time", col("avg_daily_watch_time").cast("double"))
          .withColumn("ad_interaction_rate", col("ad_interaction_rate").cast("float"))
          .drop('_rescued_data'))

# COMMAND ----------

@dlt.table(name='lal_features',
           schema = """
          ad_interaction_rate float,
          age_group string,
          avg_daily_watch_time double,
          binge_watching string,
          content_sharing string,
          education_level string,
          email string,
          ethnicity string,
          favorite_genre string,
          gender string,
          home_ownership string,
          household_income string,
          language_preference string,
          last_login_date date,
          location string,
          marital_status string,
          name string,
          num_children int,
          occupation string,
          preferred_viewing_time string,
          primary_device string,
          subscription_length double,
          subscription_type string,
          ad_interaction_percentile double,
          avg_daily_watch_percentile double,
          subscription_length_percentile double,
          user_id string not null primary key,
          days_since_last_session int,
          is_churn int,
          monthly_subscription_revenue double""")

def lal_features():
  return (dlt.read("audience_data_silver")
          .withColumn('ad_interaction_percentile', percent_rank().over(Window.orderBy("ad_interaction_rate")))
          .withColumn('avg_daily_watch_percentile', percent_rank().over(Window.orderBy("avg_daily_watch_time")))
          .withColumn('subscription_length_percentile', percent_rank().over(Window.orderBy("subscription_length")))
          .withColumn('days_since_last_session', abs(date_diff('last_login_date', current_date())))
          .withColumn('is_churn', when(col('days_since_last_session') > 14, 1).otherwise(0))
          .withColumn('monthly_subscription_revenue',
    when(col('subscription_type') == 'Premium', 20)
    .when(col('subscription_type') == 'Basic', 6.99)
    .when(col('subscription_type') == 'Ftandard', 12.99)
    .when(col('subscription_type') == 'Free Trial', 0)
    .otherwise(0)
          ))

# COMMAND ----------

@dlt.table(name='kpi_data')

def kpi_data():
  churned_users = dlt.read('lal_features').filter(col('is_churn') == 1).count()
  revenue = dlt.read('lal_features').agg(sum('monthly_subscription_revenue')).collect()[0][0]
  ad_rev = dlt.read('ad_impressions').agg(sum('WinningPrice')).collect()[0][0]

  schema = StructType(
    [
      StructField("churned_users", IntegerType(), True),
      StructField("sub_revenue", FloatType(), True),
      StructField("ad_revenue", FloatType(), True)
    ])
  
  return(
    spark.createDataFrame([(churned_users, revenue, ad_rev)],schema)
    )