# Databricks notebook source
# MAGIC %pip install dash
# MAGIC %pip install mlflow==2.17

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Imports
# prep
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, FloatType, DoubleType, LongType
# ml
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
import mlflow
from mlflow.models import infer_signature
# viz
import plotly.express as px
import plotly.graph_objects as go

# COMMAND ----------

training_df = spark.read.table('sb.ads.audience_data')

seed_id = (training_df
           .limit(10000000)
           .filter("favorite_genre == 'Comedy' or favorite_genre == 'Documentary' or favorite_genre == 'Reality TV'")
           .filter('avg_daily_watch_time > "60"')
           .filter("last_login_date >= '2024-09-01'")
           .withColumnRenamed('user_id','user_id_seed')
           .select("user_id_seed"))

print(seed_id.count())

training_df = (
    training_df
    .join(seed_id, training_df.user_id==seed_id.user_id_seed, "left_outer")
    .withColumn("label", when(seed_id.user_id_seed.isNotNull(), lit(1)).otherwise(lit(0)))
    .withColumn('last_login_date', unix_timestamp(col('last_login_date')))
    .drop('user_id_seed')
)

print(training_df.count())

# COMMAND ----------

# DBTITLE 1,Train Split
train, test = training_df.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

# DBTITLE 1,Cat Cols
cat_col = [col.name for col in training_df.schema if (isinstance(col.dataType, StringType)) and col.name not in ['user_id']]

indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid='keep') 
            for col in cat_col]

encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec", handleInvalid='keep')
            for col in cat_col]

# COMMAND ----------

# DBTITLE 1,Num Cols
num_col = [
    col.name for col in training_df.schema 
    if (isinstance(col.dataType, IntegerType) or isinstance(col.dataType, FloatType) or isinstance(col.dataType, DoubleType) or isinstance(col.dataType, LongType))
    and col.name not in ['user_id', 'label']]

numAssembler = VectorAssembler(inputCols=num_col, outputCol='num_features', handleInvalid='keep')

# COMMAND ----------

# DBTITLE 1,Pipeline
inputs = [f"{col}_vec" for col in cat_col] + ['num_features']

assembler = VectorAssembler(
    inputCols=inputs,
    outputCol="features",
)

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

pipeline = Pipeline(stages=indexers + encoders + [numAssembler] + [assembler] + [gbt])

model = pipeline.fit(train)

# COMMAND ----------

# DBTITLE 1,Predictions
predictions = (model
               .transform(training_df)
               .select('user_id','label','prediction','probability', 'age_group', 'favorite_genre','avg_daily_watch_time'))

get_positive_prob = udf(lambda item: float(item[1]), DoubleType())

predictions = (predictions.withColumn("probability", get_positive_prob("probability"))
              .select('user_id','label','prediction',round('probability', 2).alias("probability"), 'age_group', 'favorite_genre','avg_daily_watch_time'))

display(predictions)

# COMMAND ----------

# DBTITLE 1,Semi-Supervised Dataset
unknown_df = (spark.read.table('sb.ads.audience_data')
              .withColumn('last_login_date', col('last_login_date').cast('long'))
              .limit(30000000))

get_positive_prob = udf(lambda item: float(item[1]), DoubleType())

unknown_predictions = (model.transform(unknown_df))

unknown_predictions = (unknown_predictions.withColumn("probability", get_positive_prob("probability")))

unknown_predictions = (unknown_predictions.withColumnRenamed('prediction','label'))
                       
unknown_predictions = unknown_predictions.select('user_id','subscription_type', 'avg_daily_watch_time','favorite_genre','primary_device','age_group','location','last_login_date','email','name','gender','binge_watching','preferred_viewing_time','ad_interaction_rate','subscription_length','content_sharing','household_income','education_level','occupation','marital_status','num_children','ethnicity','language_preference','home_ownership','label')

# COMMAND ----------

# DBTITLE 1,Prep new training set
col_list = [col for col in training_df.columns]

unknown_predictions = unknown_predictions.select([col for col in unknown_predictions.columns if col in col_list])

# COMMAND ----------

# DBTITLE 1,Train Semi-supervised
semi_supervised = unknown_predictions.unionByName(training_df, allowMissingColumns=True).fillna(0)

train, test = semi_supervised.randomSplit([0.7, 0.3], seed=1)

model_2 = pipeline.fit(train)

new_preds = model.transform(test)

# COMMAND ----------

# DBTITLE 1,Evaluation
evaluator = MulticlassClassificationEvaluator()
evaluator.setPredictionCol('prediction')
evaluator.setProbabilityCol('probability')

# Eval Metrics
accuracy = evaluator.evaluate(new_preds, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(new_preds, {evaluator.metricName: "f1"})
weightedPrecision = evaluator.evaluate(new_preds, {evaluator.metricName: "weightedPrecision"})
weightedRecall = evaluator.evaluate(new_preds, {evaluator.metricName: "weightedRecall"})

# Confusion Matrix
predictionAndLabels = new_preds.select("prediction", "label")
metrics = MulticlassMetrics(predictionAndLabels.rdd.map(tuple))
confusion_matrix = metrics.confusionMatrix().toArray()

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Weighted Precision: {weightedPrecision:.4f}")
print(f"Weighted Recall: {weightedRecall:.4f}")
print(f"Confusion Matrix: {confusion_matrix}")

# COMMAND ----------

# DBTITLE 1,Log model metrics and register
# this might not be supported in cleanrooms today
import mlflow
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec
mlflow.set_experiment("/Users/sammy.brimberry@databricks.com/Ads-Lookalike")

with mlflow.start_run():
  mlflow.log_metrics({
    "Accuracy": accuracy,
    "F1": f1,
    "Weighted Precision": weightedPrecision,
    "Weighted Recall": weightedRecall
    })

  mlflow.set_tag("model_type", "classification")
  mlflow.set_tag("team", "data-collaboration")
  mlflow.set_tag('training info', 'semi-supervised classifcation problem for audience targeting')

  
  input_schema = train.limit(5)
  output_schema = model_2.transform(train).select('prediction')
  signature = infer_signature(input_schema, output_schema)

  mlflow.spark.log_model(model_2,
                  artifact_path = 'model_artifacts',
                  signature=signature,
                  registered_model_name="sb.ads.lookalike_model")

# COMMAND ----------

def drop_index_and_vec(df: DataFrame):
    from pyspark.sql import DataFrame
    import re
   
    all_columns = df.columns
    
    # Create a regular expression pattern to match column names ending with 'index' or 'vec'
    pattern = re.compile(r'.*(?:index|vec)$', re.IGNORECASE)
    
    columns_to_drop = [col for col in all_columns if pattern.match(col)]
    
    return df.drop(*columns_to_drop)

new_preds = drop_index_and_vec(new_preds).withColumn('probability', get_positive_prob('probability'))

(new_preds
 .write
 .mode("overwrite")
 .saveAsTable("sb.ads.lookalike_predictions"))