# Databricks notebook source
# Load the model using an alias
model_uri = f"models:/{catalog}.{schema}.lookalikemodel@Champion"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Predict on the audience data
predictions = loaded_model.predict_proba(audience_data)[:,1]
audience_data['predictions'] = predictions

display(audience_data)