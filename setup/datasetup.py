# Databricks notebook source
# MAGIC %pip install faker

# COMMAND ----------

# DBTITLE 1,Imports
from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

fake = Faker()
Faker.seed(10)
random.seed(10)
np.random.seed(10)

# COMMAND ----------

# MAGIC %run ./widgets

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auidence List Data for Lookalike Model

# COMMAND ----------

# DBTITLE 1,Random Data
# # Initialize Faker
# fake = Faker()
# Faker.seed(10)
# random.seed(10)

# # Number of records to generate
# num_records = 500000

# # Create lists to store data
# user_ids = []
# subscription_types = []
# watch_times = []
# favorite_genres = []
# device_types = []
# age_groups = []
# locations = []
# last_login_dates = []
# gender = []
# email = []
# name = []

# # Define Paramount-specific data
# subscription_options = ['Basic', 'Standard', 'Premium', 'Free Trial']
# genres = ['Drama', 'Comedy', 'Action', 'Sci-Fi', 'Documentary', 'Reality TV']
# devices = ['Smart TV', 'Mobile', 'Tablet', 'Desktop', 'Gaming Console']
# age_ranges = ['18-24', '25-34', '35-44', '45-54', '55+']
# genders = ['M','F','Other']


# # Generate data
# for _ in range(num_records):
#     user_ids.append(fake.uuid4())
#     subscription_types.append(random.choice(subscription_options))
#     watch_times.append(round(random.uniform(0, 300), 2))  # in minutes per day
#     favorite_genres.append(random.choice(genres))
#     device_types.append(random.choice(devices))
#     age_groups.append(random.choice(age_ranges))
#     locations.append(fake.state())
#     last_login_dates.append(fake.date_between(start_date='-30d', end_date='today'))
#     email.append(fake.ascii_email())
#     name.append(fake.name())
#     gender.append(random.choice(genders))

# binge_watching = [random.choice(['Frequent', 'Occasional', 'Rare', 'Never']) for _ in range(num_records)]
# preferred_viewing_time = [random.choice(['Morning', 'Afternoon', 'Evening', 'Late Night']) for _ in range(num_records)]
# ad_interaction_rate = [round(random.uniform(0, 1), 2) for _ in range(num_records)]
# subscription_length = [random.randint(1, 60) for _ in range(num_records)]  # in months
# content_sharing = [random.choice(['High', 'Medium', 'Low', 'None']) for _ in range(num_records)]
# household_income = [random.choice(['<$25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '$100k-$150k', '>$150k']) for _ in range(num_records)]
# education_level = [random.choice(['High School', 'Some College', "Bachelor's", "Master's", 'Doctorate']) for _ in range(num_records)]
# occupation = [fake.job() for _ in range(num_records)]
# marital_status = [random.choice(['Single', 'Married', 'Divorced', 'Widowed']) for _ in range(num_records)]
# num_children = [random.choice([0, 1, 2, 3, '4+']) for _ in range(num_records)]
# ethnicity = [random.choice(['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other']) for _ in range(num_records)]
# language_preference = [random.choice(['English', 'Spanish', 'French', 'Mandarin', 'Hindi']) for _ in range(num_records)]
# home_ownership = [random.choice(['Own', 'Rent', 'Other']) for _ in range(num_records)]

# # Create DataFrame
# df = pd.DataFrame({
#     'user_id': user_ids,
#     'subscription_type': subscription_types,
#     'avg_daily_watch_time': watch_times,
#     'favorite_genre': favorite_genres,
#     'primary_device': device_types,
#     'age_group': age_groups,
#     'location': locations,
#     'last_login_date': last_login_dates,
#     'email': email,
#     'name': name,
#     'gender': gender,
#     "binge_watching" : binge_watching,
#     "preferred_viewing_time" : preferred_viewing_time,
#     "ad_interaction_rate" : ad_interaction_rate,
#     "subscription_length" : subscription_length,
#     "content_sharing" : content_sharing,
#     "household_income" : household_income,
#     "education_level" : education_level,
#     "occupation" : occupation,
#     "marital_status" : marital_status,
#     "num_children" : num_children,
#     "ethnicity" : ethnicity,
#     "language_preference" : language_preference,
#     "home_ownership" : home_ownership
# })

# COMMAND ----------

# DBTITLE 1,Faux Patterns
# Initialize Faker
fake = Faker()
Faker.seed(10)
random.seed(10)
np.random.seed(10)

num_records = 100000

subscription_options = ['Basic', 'Standard', 'Premium', 'Free Trial']
genres = ['Drama', 'Comedy', 'Action', 'Sci-Fi', 'Documentary', 'Reality TV']
devices = ['Smart TV', 'Mobile', 'Tablet', 'Desktop', 'Gaming Console']
age_ranges = ['18-24', '25-34', '35-44', '45-54', '55+']
genders = ['M', 'F', 'Other']
binge_options = ['Frequent', 'Occasional', 'Rare', 'Never']
viewing_times = ['Morning', 'Afternoon', 'Evening', 'Late Night']
sharing_options = ['High', 'Medium', 'Low', 'None']
income_brackets = ['<$25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '$100k-$150k', '>$150k']
education_levels = ['High School', 'Some College', "Bachelor's", "Master's", 'Doctorate']
marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
child_options = [0, 1, 2, 3, 4]
ethnicities = ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other']
languages = ['English', 'Spanish', 'French', 'Mandarin', 'Hindi']
home_ownership_options = ['Own', 'Rent', 'Other']

def get_watch_time(subscription):
    if subscription == 'Premium':
        return max(0, min(300, np.random.normal(180, 30)))
    elif subscription == 'Standard':
        return max(0, min(300, np.random.normal(120, 30)))
    elif subscription == 'Basic':
        return max(0, min(300, np.random.normal(90, 30)))
    else:  # Free Trial
        return max(0, min(300, np.random.normal(60, 45)))

def get_favorite_genre(age_group):
    if age_group in ['18-24', '25-34']:
        return np.random.choice(genres, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
    elif age_group in ['35-44', '45-54']:
        return np.random.choice(genres, p=[0.2, 0.2, 0.2, 0.1, 0.2, 0.1])
    else:  # 55+
        return np.random.choice(genres, p=[0.3, 0.1, 0.1, 0.1, 0.3, 0.1])

def get_device(age_group):
    if age_group in ['18-24', '25-34']:
        return np.random.choice(devices, p=[0.2, 0.4, 0.1, 0.1, 0.2])
    elif age_group in ['35-44', '45-54']:
        return np.random.choice(devices, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    else:  # 55+
        return np.random.choice(devices, p=[0.4, 0.1, 0.1, 0.3, 0.1])

def get_subscription_length(subscription):
    if subscription == 'Premium':
        return random.randint(12, 60)
    elif subscription == 'Standard':
        return random.randint(6, 36)
    elif subscription == 'Basic':
        return random.randint(3, 24)
    else:  # Free Trial
        return random.randint(1, 3)

def get_binge_watching(watch_time):
    if watch_time > 200:
        return np.random.choice(binge_options, p=[0.6, 0.3, 0.05, 0.05])
    elif watch_time > 100:
        return np.random.choice(binge_options, p=[0.3, 0.4, 0.2, 0.1])
    else:
        return np.random.choice(binge_options, p=[0.1, 0.2, 0.4, 0.3])

def get_viewing_time(age_group):
    if age_group in ['18-24', '25-34']:
        return np.random.choice(viewing_times, p=[0.1, 0.2, 0.3, 0.4])
    elif age_group in ['35-44', '45-54']:
        return np.random.choice(viewing_times, p=[0.2, 0.2, 0.4, 0.2])
    else:  # 55+
        return np.random.choice(viewing_times, p=[0.3, 0.3, 0.3, 0.1])

def get_subscription(income):
    if income in ['$100k-$150k', '>$150k']:
        return np.random.choice(subscription_options, p=[0.1, 0.2, 0.6, 0.1])
    elif income in ['$75k-$100k', '$50k-$75k']:
        return np.random.choice(subscription_options, p=[0.2, 0.4, 0.3, 0.1])
    else:
        return np.random.choice(subscription_options, p=[0.4, 0.3, 0.1, 0.2])

data = []
for _ in range(num_records):
    age_group = random.choice(age_ranges)
    income = random.choice(income_brackets)
    subscription = get_subscription(income)
    watch_time = get_watch_time(subscription)
    marital_status = random.choice(marital_statuses)
    num_children = random.choice(child_options)
    
    if marital_status == 'Married' and num_children != 0:
        watch_time *= 0.8  # Reduce watch time for married with children

    record = {
        'user_id': fake.uuid4(),
        'subscription_type': subscription,
        'avg_daily_watch_time': round(watch_time, 2),
        'favorite_genre': get_favorite_genre(age_group),
        'primary_device': get_device(age_group),
        'age_group': age_group,
        'location': fake.state(),
        'last_login_date': fake.date_between(start_date='-30d', end_date='today'),
        'email': fake.ascii_email(),
        'name': fake.name(),
        'gender': random.choice(genders),
        'binge_watching': get_binge_watching(watch_time),
        'preferred_viewing_time': get_viewing_time(age_group),
        'ad_interaction_rate': round(random.uniform(0, 1), 2),
        'subscription_length': get_subscription_length(subscription),
        'content_sharing': random.choice(sharing_options),
        'household_income': income,
        'education_level': random.choice(education_levels),
        'occupation': fake.job(),
        'marital_status': marital_status,
        'num_children': num_children,
        'ethnicity': random.choice(ethnicities),
        'language_preference': random.choice(languages),
        'home_ownership': random.choice(home_ownership_options)
    }
    data.append(record)

# Create DataFrame
audience_df = pd.DataFrame(data)

# Additional correlation: Education level and favorite genre
audience_df.loc[audience_df['education_level'].isin(["Master's", 'Doctorate']) & (audience_df['favorite_genre'] != 'Documentary'), 'favorite_genre'] = np.random.choice(genres, size=len(audience_df[audience_df['education_level'].isin(["Master's", 'Doctorate']) & (audience_df['favorite_genre'] != 'Documentary')]), p=[0.2, 0.1, 0.1, 0.1, 0.4, 0.1])

# Correlation between ethnicity and language preference
ethnicity_language_map = {
    'Hispanic': 'Spanish',
    'Asian': 'Mandarin',
    'Caucasian': 'English',
    'African American': 'English',
    'Other': 'English'
}
audience_df['language_preference'] = audience_df.apply(lambda row: ethnicity_language_map[row['ethnicity']] if random.random() < 0.7 else row['language_preference'], axis=1)

audience_df = spark.createDataFrame(audience_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seed File Dataset

# COMMAND ----------

seed_df = (
    audience_df
    .filter("favorite_genre == 'Comedy' or favorite_genre == 'Documentary' or favorite_genre == 'Reality TV'")
    .filter("subscription_type == 'Free Trial' or subscription_type == 'Basic' or subscription_type == 'Standard'")
    .filter('avg_daily_watch_time > "60"')
    .filter(f"last_login_date >= '{(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')}'")
    .withColumnRenamed('user_id', 'user_id_seed')
    .select("user_id_seed")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ad Impression Data

# COMMAND ----------

Faker.seed(12345)
random.seed(12345)

num_records = 10000

data = []

shows = ['Star Trek: Discovery', 'Yellowstone', 'The Good Fight', 'South Park', 'SpongeBob SquarePants']
movies = ['Top Gun: Maverick', 'Mission: Impossible', 'Transformers', 'The Godfather', 'Titanic']
ad_types = ['Pre-roll', 'Mid-roll', 'Post-roll', 'Overlay', 'Interactive']
device_types = ['Smart TV', 'Mobile', 'Tablet', 'Desktop', 'Gaming Console']
subscription_types = ['Basic', 'Standard', 'Premium', 'Free Trial']

for _ in range(num_records):
    timestamp = fake.date_time_between(start_date='-30d', end_date='now')
    
    record = {
        'Timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'ImpressionID': fake.uuid4(),
        'CampaignID': f'CAMP{fake.random_number(digits=6, fix_len=True)}',
        'CreativeID': f'CREA{fake.random_number(digits=6, fix_len=True)}',
        'PublisherID': 'PARAMOUNT',
        'PlacementID': f'PLAC{fake.random_number(digits=6, fix_len=True)}',
        'UserID': fake.uuid4(),
        'DeviceType': random.choice(device_types),
        'IPAddress': fake.ipv4(),
        'GeographicLocation': fake.country_code(),
        'AdUnitSize': random.choice(['1920x1080', '1280x720', '640x360', '320x180']),
        'BidPrice': round(random.uniform(1, 20), 2),
        'WinningPrice': round(random.uniform(1, 20), 2),
        'ViewabilityScore': round(random.uniform(0.5, 1), 2),
        'CompletionRate': round(random.uniform(0.1, 1), 2),
        'ClickThrough': random.choice([0, 1]),
        'AudienceSegmentID': f'SEG{fake.random_number(digits=4, fix_len=True)}',
        'ContentID': random.choice(shows + movies),
        'ContentType': 'Show' if random.random() < 0.7 else 'Movie',
        'AdBreakID': f'BREAK{fake.random_number(digits=6, fix_len=True)}',
        'PodPosition': random.randint(1, 5),
        'AdType': random.choice(ad_types),
        'SubscriptionType': random.choice(subscription_types),
        'AdDuration': random.choice([15, 30, 60]),
        'UserAge': random.randint(18, 80),
        'UserGender': random.choice(['Male', 'Female', 'Other']),
    }
    data.append(record)

ad_impressions = pd.DataFrame(data)

ad_impressions = spark.createDataFrame(ad_impressions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write data to volume

# COMMAND ----------

to_write = {
    'ad_impressions': ad_impressions,
    'audience_df': audience_df,
    'seed_df': seed_df
}

for name, data in to_write.items():
  data.repartition(100).write.mode('append').format('json').save(folder + f'/{name}')

# COMMAND ----------

def cleanup_folder(path):
  #Cleanup to have something nicer
  for f in dbutils.fs.ls(path):
    if f.name.startswith('_committed') or f.name.startswith('_started') or f.name.startswith('_SUCCESS') :
      dbutils.fs.rm(f.path)

for name, data in to_write.items():
  cleanup_folder(folder + f'/{name}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Expectations

# COMMAND ----------

# MAGIC %sql
# MAGIC -- drop table IDENTIFIER(:catalog || '.' || :schema || '.quality_rules');
# MAGIC create or replace table IDENTIFIER(:catalog || '.' || :schema || '.quality_rules') 
# MAGIC as select
# MAGIC   col1 AS name,
# MAGIC   col2 AS constraint,
# MAGIC   col3 AS tag
# MAGIC from (
# MAGIC   values
# MAGIC   ('email_not_null', 'email IS NOT NULL', 'audience_team'),
# MAGIC   ('email_format_valid', 'email RLIKE "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$"', 'audience_team'),
# MAGIC   ('gender_valid', 'gender IN ("M", "F", "Other")', 'audience_team')
# MAGIC ) as t(col1, col2, col3);