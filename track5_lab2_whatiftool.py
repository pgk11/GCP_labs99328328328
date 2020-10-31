import pandas as pd
import xgboost as xgb
import numpy as np
import collections
import witwidget

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

# Use a small subset of the data since the original dataset is too big for Colab (2.5GB)
# Data source: https://www.ffiec.gov/hmda/hmdaflat.htm
!gsutil cp gs://mortgage_dataset_files/mortgage-small.csv .

# Set column dtypes for Pandas
COLUMN_NAMES = collections.OrderedDict({
  'as_of_year': np.int16,
  'agency_code': 'category',
  'loan_type': 'category',
  'property_type': 'category',
  'loan_purpose': 'category',
  'occupancy': np.int8,
  'loan_amt_thousands': np.float64,
  'preapproval': 'category',
  'county_code': np.float64,
  'applicant_income_thousands': np.float64,
  'purchaser_type': 'category',
  'hoepa_status': 'category',
  'lien_status': 'category',
  'population': np.float64,
  'ffiec_median_fam_income': np.float64,
  'tract_to_msa_income_pct': np.float64,
  'num_owner_occupied_units': np.float64,
  'num_1_to_4_family_units': np.float64,
  'approved': np.int8
})

# Load data into Pandas
data = pd.read_csv(
  'mortgage-small.csv', 
  index_col=False,
  dtype=COLUMN_NAMES
)
data = data.dropna()
data = shuffle(data, random_state=2)
data.head()

# Label preprocessing
labels = data['approved'].values

# See the distribution of approved / denied classes (0: denied, 1: approved)
print(data['approved'].value_counts())

data = data.drop(columns=['approved'])

# Convert categorical columns to dummy columns
dummy_columns = list(data.dtypes[data.dtypes == 'category'].index)
data = pd.get_dummies(data, columns=dummy_columns)

# Preview the data
data.head()

# Split the data into train / test sets
x,y = data,labels
x_train,x_test,y_train,y_test = train_test_split(x,y)

# Train the model, this will take a few minutes to run
bst = xgb.XGBClassifier(
    objective='reg:logistic'
)

bst.fit(x_train, y_train)

# Get predictions on the test set and print the accuracy score
y_pred = bst.predict(x_test)
acc = accuracy_score(y_test, y_pred.round())
print(acc, '\n')

# Print a confusion matrix
print('Confusion matrix:')
cm = confusion_matrix(y_test, y_pred.round())
cm = cm / cm.astype(np.float).sum(axis=1)
print(cm)

# Save the model so we can deploy it
bst.save_model('model.bst')

#Deploy model to AI Platform

GCP_PROJECT = 'YOUR_GCP_PROJECT'
MODEL_BUCKET = 'gs://your_storage_bucket'
MODEL_NAME = 'your_model_name' # You'll create this model below
VERSION_NAME = 'v1'

# Copy your model file to Cloud Storage
!gsutil cp ./model.bst $MODEL_BUCKET

# Configure gcloud to use your project
!gcloud config set project $GCP_PROJECT

# Create a model
!gcloud ai-platform models create $MODEL_NAME --regions us-central1

# Create a version, this will take ~2 minutes to deploy
!gcloud ai-platform versions create $VERSION_NAME \
--model=$MODEL_NAME \
--framework='XGBOOST' \
--runtime-version=1.15 \
--origin=$MODEL_BUCKET \
--staging-bucket=$MODEL_BUCKET \
--python-version=3.7 \
--project=$GCP_PROJECT

# Format a subset of the test data to send to the What-if Tool for visualization
# Append ground truth label value to training data

# This is the number of examples you want to display in the What-if Tool
num_wit_examples = 500
test_examples = np.hstack((x_test[:num_wit_examples].values,y_test[:num_wit_examples].reshape(-1,1)))

# Create a What-if Tool visualization, it may take a minute to load
# See the cell below this for exploration ideas

# This prediction adjustment function is needed as this xgboost model's
# prediction returns just a score for the positive class of the binary
# classification, whereas the What-If Tool expects a list of scores for each
# class (in this case, both the negative class and the positive class).
def adjust_prediction(pred):
  return [1 - pred, pred]

config_builder = (WitConfigBuilder(test_examples.tolist(), data.columns.tolist() + ['mortgage_status'])
  .set_ai_platform_model(GCP_PROJECT, MODEL_NAME, VERSION_NAME, adjust_prediction=adjust_prediction)
  .set_target_feature('mortgage_status')
  .set_label_vocab(['denied', 'approved']))
WitWidget(config_builder, height=800)

