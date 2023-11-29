# Marketing strategy using Data Science on website attribution data

## Setup

For the task, I have used poetry to create the virtual environment.

1.Install poetry

   ```bash
   pip3 install poetry
   ```

2.Goto the directory where `pyproject.toml` is located:
eg- `Senior Data Scientist Task` where `pyproject.toml` is located
Run:

   ```bash
   poetry install
   ```

Use the virtual environment created above to run the jupyter notebook below.

## Datasets

The datasets containing information about a sample of the users who have visited the website in the past.User activity on the site is grouped into sessions (see web session definition for more details) and sessions are grouped together into user paths (see path exploration: cross-session for more details).The data includes aggregated user interaction features (e.g.the number of sessions in a path, pages a user has seen, interactions with filters, etc.) on the website across all of a users’ sessions, as well as sequential data detailing the marketing channels through which users come to the site.

These are examples of some of the features  have available to us when predicting how likely a user is to book on the site. would like you to build a machine learning model to predict the likelihood of users booking on the site based on the information provided about their historical path on the website so far.

Two datasets:

### attribution_path_data

Contains sequential information about how a customer has accessed the site.This includes the marketing channels; device types; and the time lags between sessions.This dataset also includes the `has_booked` flag to use as the target variable.

One row of `attribution_path_data` consists of a unique session that a user had within a path.One can group the data by `path_id` to find all of the sessions which occurred in that path.The columns are explained as follows:

- path_id: identifier column for which path the session is associated with.
- device_type: the label encoded device type associated with the session.
- attribution_channel: the label encoded marketing channel used to access the site for that session.
- distance_to_last: the order of the sessions.A value of 1 denotes the last session in the path, and the max distance_to_last value denotes the first session in the path.
- time_delta_in_days: the time difference in days between a given session and the last session in the path.
- has_booked: the target variable which determines if a path ended with a booking.All sessions in a booked_path will have a value of 1 for this column.

### user_feature_data

Contains aggregated information across all sessions in a user’s path detailing information about a user’s activity on the website. In this dataset  have one row per path, where the features have been aggregated for a path.

The columns are explained as follows:

- path_id: the unique path identifier.
- n_sessions: the number of sessions in the path.
- most_common_landing_page: the most common landing page in the path.
- clicked_x: whether a user clicked on a given website feature x (e.g.clicked_beach - the user clicked on a holiday to a beach destination).
- viewed_x: whether a user viewed a given website feature x (e.g.viewed_beach).
- saw_x: whether a user saw a given funnel page on the website (e.g.saw the price and availability page saw_panda).
- adults: the average value inputted into the adults filter on the website.
- children: the average value inputted into the children filter on the website.
- nights: the average value inputted into the nights filter on the website.

## EDA

The `eda.ipynb` notebook analyzes a dataset to understand its characteristics and patterns. It starts with an overview of the dataset and its variables, followed by data cleaning to ensure the data's quality.Visualization techniques and statistical analysis are used to explore the data and identify relationships between variables.The notebook concludes by summarizing the findings and providing insights for further analysis and modeling.

## Pipeline

The `full_pipeline.ipynb` notebook focuses on building a complete machine learning pipeline from data preparation to model training and evaluation.

- It starts with data loading and preprocessing, including data cleaning, feature engineering, and normalization.
- Train multiple binary classification models using different algorithms .
- Evaluate each model on the validation set.
- Select the best model based on the validation set metrics.
- Make predictions on the new data using the best model.
- Use the test set metrics to set thresholds for retraining the model.
- The notebook concludes by summarizing the pipeline's performance and providing insights for improvement.
