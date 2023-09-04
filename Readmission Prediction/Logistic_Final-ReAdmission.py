#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image


# In[19]:


import os
import pandas as pd
import numpy as np

class ReadmissionReader:
    def __init__(self, root_dir, mode='train', seed=42):
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'"
        self.root_dir = root_dir
        self.mode_dir = os.path.join(root_dir, mode)
        self.seed = seed
        self.patient_files = self._get_patient_files()
        np.random.seed(seed)
        np.random.shuffle(self.patient_files)

    def _get_patient_files(self):
        patient_files = []
        for patient in os.listdir(self.mode_dir):
            patient_dir = os.path.join(self.mode_dir, patient)
            if os.path.isdir(patient_dir):
                patient_files.append(patient_dir)
        return patient_files

    def get_number_of_examples(self):
        return len(self.patient_files)

    def read_example(self, i):
        patient_dir = self.patient_files[i]
        patient_id = os.path.basename(patient_dir)
        episode_files = [f for f in os.listdir(patient_dir) if f.startswith('episode')]
        episode_timeseries_files = [f for f in episode_files if f.endswith('_timeseries.csv')]
        episode_timeseries_files.sort()

        # Check if there are any episode timeseries files
        if not episode_timeseries_files:
            return None

        # Load the first episode timeseries data
        episode1_timeseries_file = episode_timeseries_files[0]
        episode1_timeseries_path = os.path.join(patient_dir, episode1_timeseries_file)
        episode1_timeseries_data = pd.read_csv(episode1_timeseries_path)
        episode1_timeseries_data = episode1_timeseries_data.fillna('')
        # Get the header
        header = episode1_timeseries_data.columns.tolist()

        # Check if there are more than one episode timeseries files
        y_label = 1 if len(episode_timeseries_files) > 1 else 0

        # Get the total time
        total_time = float(episode1_timeseries_data.iloc[-1, 0])

        example = {
            'X': episode1_timeseries_data.values.astype(str),
            't': total_time,
            'y': y_label,
            'header': header,
            'name': f"{patient_id}_{episode1_timeseries_file}"
        }

        return example


# In[20]:


root_dir = r'C:\Users\samit\OneDrive\Desktop\Datasets\data\root'


# In[21]:


train_reader = ReadmissionReader(root_dir, mode='train')
test_reader = ReadmissionReader(root_dir, mode='test')


# In[22]:


global_category_mapping = {
  'Glascow coma scale eye opening': {'': 0,
  'To Pain': 1,
  'None': 2,
  'Spontaneously': 3,
  'To Speech': 4,
  '4 Spontaneously': 5,
  '1 No Response': 6,
  '2 To pain': 7,
  '3 To speech': 8},
 'Glascow coma scale motor response': {'': 0,
  'Localizes Pain': 1,
  '3 Abnorm flexion': 2,
  'Abnormal extension': 3,
  '5 Localizes Pain': 4,
  '6 Obeys Commands': 5,
  'Flex-withdraws': 6,
  '2 Abnorm extensn': 7,
  'Obeys Commands': 8,
  'Abnormal Flexion': 9,
  'No response': 10,
  '4 Flex-withdraws': 11,
  '1 No Response': 12},
 'Glascow coma scale verbal response': {'': 0,
  'Inappropriate Words': 1,
  'Incomprehensible sounds': 2,
  'Oriented': 3,
  '2 Incomp sounds': 4,
  '4 Confused': 5,
  'No Response-ETT': 6,
  '3 Inapprop words': 7,
  'Confused': 8,
  'No Response': 9,
  '1.0 ET/Trach': 10,
  '5 Oriented': 11,
  '1 No Response': 12}}


# In[23]:


import numpy as np
import pandas as pd

TIME_WINDOWS = [(0, 6), (0, 12), (0, '25%'), (0, '50%'), (0, '100%')]

CHANNELS = {
    'Diastolic blood pressure': 'numerical',
    'Fraction inspired oxygen': 'numerical',
    'Glascow coma scale eye opening': 'categorical',
    'Glascow coma scale motor response': 'categorical',
    'Glascow coma scale total': 'numerical',
    'Glascow coma scale verbal response': 'categorical',
    'Glucose': 'numerical',
    'Heart Rate': 'numerical',
    'Mean blood pressure': 'numerical',
    'Oxygen saturation': 'numerical',
    'Respiratory rate': 'numerical',
    'Systolic blood pressure': 'numerical',
    'Temperature': 'numerical',
    'Weight': 'numerical'
}

admission_mapping = {'UNKNOWN': 0, 'EMERGENCY': 1, 'ELECTIVE': 2, 'URGENT': 3}
dbsource_mapping = {'carevue': 1, 'metavision': 2, 'both': 3}
icu_unit_mapping = {'MICU': 1, 'CCU': 2, 'TSICU': 3, 'SICU': 4, 'CSRU': 5}

def extract_features_from_reader(train_reader, index, root):
     # Read the example data from the reader
    example_data = train_reader.read_example(index)
    if example_data is None:
        return None, None
    try: 
        ts_data = pd.DataFrame(data=example_data['X'], columns=example_data['header'])

    #     Replace empty strings with NaN
    #     ts_data = ts_data.replace('', np.nan)

    #     # Forward fill to carry forward the last known value
    #     ts_data = ts_data.fillna(method='ffill')

    #     # Backward fill to fill any remaining NaNs
    #     ts_data = ts_data.fillna(method='bfill')

    #     # If there are still any NaNs, fill them with 0
    #     ts_data = ts_data.fillna(0)
    #     ts_data = ts_data.apply(pd.to_numeric, errors='coerce')
        ts_data['Hours'] = ts_data['Hours'].astype(float)

        all_features = []
        all_categorical_features = []
        labels = []
        for start_hour, end_hour in TIME_WINDOWS:
            # Determine the end_hour for percentage-based windows
            if isinstance(start_hour, str):
                start_hour = ts_data['Hours'].quantile(0.75)
            if isinstance(end_hour, str):
                end_hour = ts_data['Hours'].max()
    #             total_hours = ts_data['Hours'].max()
    #             end_hour = total_hours * float(end_hour.strip('%')) / 100.0



            # Filter the data for the current time window
            ts_window_data = ts_data[(ts_data['Hours'] >= start_hour) & (ts_data['Hours'] <= end_hour)]
            ts_window_data_num = ts_window_data.apply(pd.to_numeric, errors='coerce')
            features = []
            for channel, channel_type in CHANNELS.items():
                if channel_type == 'numerical'and channel in list(CHANNELS.keys()):
                    # Debug: Check if mean_value is NaN

                    # Extract statistics for numerical channels
                    features.append(ts_window_data_num[channel].mean())
                    features.append(ts_window_data_num[channel].std())
                    features.append(ts_window_data_num[channel].min())
                    features.append(ts_window_data_num[channel].max())
                    features.append(ts_window_data_num[channel].skew())
    #                 features.append(ts_window_data[channel].iloc[-1] if not ts_window_data[channel].empty else np.nan)
                    features.append(len(ts_window_data_num[channel]))
                    features.append(ts_window_data_num[channel].quantile(0.25))
                    features.append(ts_window_data_num[channel].quantile(0.75))
                    features.append(ts_window_data_num[channel].quantile(0.75) - ts_window_data_num[channel].quantile(0.25))


                if channel_type == 'categorical':
                    # Get the value counts
                    value_counts = ts_window_data[channel].value_counts()

                    # Find the most frequent non-empty category
                    most_frequent_category = ''
                    for category in value_counts.index:
                        if category != '':
                            most_frequent_category = category
                            break

                    # If no non-empty category is found, default to zero
                    if most_frequent_category == '':
                        mapped_value = 0
                    else:
                        # Map it to its numerical value
                        mapped_value = global_category_mapping[channel].get(most_frequent_category, 0)

                    # Append it to the categorical features list
                    all_categorical_features.append(mapped_value)

            all_features.extend(features)
        all_features =  np.array(all_features)
        all_features = np.nan_to_num(all_features)
        # Convert categorical features list to NumPy array and append it to all_features
        all_categorical_features = np.array(all_categorical_features)

        # Extract the patient ID and episode number from the file name
        file_name = example_data['name']
        patient_id, episode_num = file_name.split('_')[0], file_name.split('_')[1].replace('episode', '')

        # Define the root directory and train folder
        if root == 'train':
            train_dir = os.path.join(root_dir, 'train')
        elif root == 'test':
            train_dir = os.path.join(root_dir, 'test')

        # Load the combined data
        combined_data = pd.read_csv('combined_data')

        # Access the patient folder
        patient_path = os.path.join(train_dir, patient_id)

        # Load the stays data
        stays_data = pd.read_csv(os.path.join(patient_path, 'stays.csv'))

        # Access the episode folder
        #episode_path = os.path.join(patient_path, f'episode{episode_num}')

        # Load the episode data
        episode_data = pd.read_csv(os.path.join(patient_path, f'episode{episode_num}.csv'))

        # Extract the relevant features
        age = episode_data['Age'].values[0]
        weight = episode_data['Weight'].values[0]
        ethnicity = episode_data['Ethnicity'].values[0]
        gender = episode_data['Gender'].values[0]
        dbsource = stays_data['DBSOURCE'].values[0]
        last_careunit = stays_data['LAST_CAREUNIT'].values[0]

        # Get the admission type from the combined data
        # Get the admission type from the combined data
        icustay_id = stays_data['ICUSTAY_ID'].values[0]
        matching_rows = combined_data[combined_data['ICUSTAY_ID'] == icustay_id]

        if not matching_rows.empty:
            admission_type = matching_rows['ADMISSION_TYPE'].values[0]
        else:
            # Handle the case where there are no matching rows
            print(f"No matching rows found for ICUSTAY_ID {icustay_id}")
            admission_type = "UNKNOWN"


        # Map the categorical variables to their numerical representations
        mapped_admission_type = admission_mapping.get(admission_type, 0)  # Default to 0 if not found
        mapped_dbsource = dbsource_mapping.get(dbsource, 0)  # Default to 0 if not found
        mapped_last_careunit = icu_unit_mapping.get(last_careunit, 0)  # Default to 0 if not found

        # Create the static data vector with the mapped values
        static_data_vector = [age, weight, gender, ethnicity, mapped_dbsource, mapped_last_careunit, mapped_admission_type]


        all_features = np.concatenate((all_features, all_categorical_features,static_data_vector))

        labels.append(example_data['y'])

        return all_features, labels 
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    
# Example usage
# all_features = extract_features_from_reader(train_reader, 0)


# In[24]:


train_features = []
train_labels = []
for i in range(train_reader.get_number_of_examples()):
    all_features, label = extract_features_from_reader(train_reader, i,'train')
    if all_features is None or label is None:
        continue
    train_features.append(all_features)
    train_labels.append(label)


# In[8]:


# val_features = []
# val_labels = []
# for i in range(val_reader.get_number_of_examples()):
#     all_features, label = extract_features_from_reader(val_reader, i,'train')
#     val_features.append(all_features)
#     val_labels.append(label)


# In[25]:


test_features = []
test_labels = []
for i in range(test_reader.get_number_of_examples()):
    all_features, label = extract_features_from_reader(test_reader, i,'test')
    if all_features is None or label is None:
        continue
    test_features.append(all_features)
    test_labels.append(label)


# In[26]:


train_features = np.array(train_features)
#val_features = np.array(val_features)
test_features = np.array(test_features)


# In[27]:


train_labels = np.array(train_labels)
#val_labels = np.array(val_labels)
test_labels = np.array(test_labels)


# In[28]:


# Define the indices of the categorical features
categorical_indices = list(range(495, 511)) + list(range(512, 518))

# Define the indices of the numerical features
numerical_indices = [i for i in range(train_features.shape[1]) if i not in categorical_indices]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both
train_features[:, numerical_indices] = scaler.fit_transform(train_features[:, numerical_indices])

# Use the same scaler to transform the validation and test data
#val_features[:, numerical_indices] = scaler.transform(val_features[:, numerical_indices])
test_features[:, numerical_indices] = scaler.transform(test_features[:, numerical_indices])


# In[29]:





# In[ ]:





# In[30]:


train_labels = train_labels.reshape(-1)
#val_labels = val_labels.reshape(-1)
test_labels = test_labels.reshape(-1)


# In[31]:


train_labels


# In[33]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(train_features)
#X_val = imputer.transform(val_features)
X_test = imputer.transform(test_features)


# In[34]:


np.save('X_train_r.npy',X_train)
#np.save('X_val_r.npy',X_val)
np.save('X_test_r.npy',X_test)
np.save('train_labels_r',train_labels)
#np.save('val_labels_r',val_labels)
np.save('test_labels_r',test_labels)


# In[35]:


X_train[0]


# In[36]:


import numpy as np

# Assume your_array is the array you want to check
num_nan = np.isnan(X_train).sum()

print(f"Total number of NaN values in the array: {num_nan}")


# In[52]:


model = LogisticRegression(penalty = 'l2',C = 0.2, class_weight = 'balanced', max_iter = 1000)
model.fit(X_train, train_labels)


# In[53]:


y_test_pred = model.predict(X_test)


# In[54]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Assuming y_val_true and y_test_true are your ground truth labels for validation and test sets respectively
# and y_val_pred and y_test_pred are the predicted labels from your model for validation and test sets respectively

# # Validation metrics
# val_accuracy = accuracy_score(val_labels, y_val_pred)
# val_precision = precision_score(val_labels, y_val_pred)
# val_recall = recall_score(val_labels, y_val_pred)
# val_f1 = f1_score(val_labels, y_val_pred)
# val_roc_auc = roc_auc_score(val_labels, y_val_pred)

# print("Validation Metrics:")
# print(f"Accuracy: {val_accuracy}")
# print(f"Precision: {val_precision}")
# print(f"Recall: {val_recall}")
# print(f"F1 Score: {val_f1}")
# print(f"ROC AUC: {val_roc_auc}")

# Test metrics
test_accuracy = accuracy_score(test_labels, y_test_pred)
test_precision = precision_score(test_labels, y_test_pred)
test_recall = recall_score(test_labels, y_test_pred)
test_f1 = f1_score(test_labels, y_test_pred)
test_roc_auc = roc_auc_score(test_labels, y_test_pred)

print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")
print(f"ROC AUC: {test_roc_auc}")


# In[63]:


# Model names and AUC-ROC values
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
model_names = ["Logistic Regression","Normal LSTM", "Bidirectional LSTM", "LSTM without Multimodality", "Channel-wise LSTM"]
micro_auc_values = [0.613,0.670, 0.614, 0.606, 0.651]
macro_auc_values = [0.624014295026788,0.65,0.64,0.66,0.64]

# Combine the data and sort by Micro and Macro AUC-ROC values
data_micro = list(zip(model_names, micro_auc_values))
data_macro = list(zip(model_names, macro_auc_values))

data_micro.sort(key=lambda x: x[1], reverse=True)
data_macro.sort(key=lambda x: x[1], reverse=True)

sorted_model_names_micro, sorted_micro_auc = zip(*data_micro)
sorted_model_names_macro, sorted_macro_auc = zip(*data_macro)

# Set up the bar positions
bar_width = 0.6
index = np.arange(len(model_names))

# Create the bars
sns.set_style("whitegrid")

# Plot for Micro AUC-ROC
plt.figure(figsize=(12, 6))
bar1 = plt.bar(index, sorted_micro_auc, bar_width,  color=sns.color_palette("pastel")[0], label='Micro AUC-ROC')
plt.xlabel('Model', fontsize=14)
plt.ylabel('AUC-ROC Value', fontsize=14)
plt.title('Comparison of AUC-ROC values across models', fontsize=16)
plt.xticks(index, sorted_model_names_micro, rotation=45, ha='right', fontsize=12)
plt.ylim(0.58, 0.68)
plt.plot(index, sorted_micro_auc, color='red', marker='o', linestyle='dashed', linewidth=1, markersize=5)
for i, v in enumerate(sorted_micro_auc):
    plt.text(i, v + 0.005, "{:.2f}".format(v), ha='center')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[55]:


# Define a function to plot the AUC-ROC curve for each model
def plot_roc_curve(model, X_val, y_val, label, color):
    y_pred = model.predict(X_val)
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    auc_value = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label='{} AUC = {:.2f}'.format(label, auc_value))


# In[56]:


# Plot the AUC-ROC curves for each model in a single plot
plt.figure(figsize=(8, 6))
plot_roc_curve(model, X_test, test_labels, 'Normal LSTM', 'blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[ ]:





# In[57]:


rf = RandomForestClassifier()
rf.fit(X_train, train_labels)


# In[58]:


y_pred_r = rf.predict(X_test)


# In[59]:


test_roc_auc_rf = roc_auc_score(test_labels, y_pred_r)


# In[60]:


print(test_roc_auc_rf)


# In[52]:


param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, train_labels)


# In[53]:


# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)


# In[54]:


y_pred_cv = best_rf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(test_labels, y_pred_cv)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();


# In[55]:


test_roc_auc_cv = roc_auc_score(test_labels, y_pred_cv)


# In[56]:


print(test_roc_auc_cv)


# In[ ]:





# In[ ]:





# In[ ]:





# In[203]:


all_features, label = extract_features_from_reader(train_reader, 0,'train')


# In[204]:


len(all_features)


# In[205]:


all_features


# In[178]:


label


# In[ ]:





# In[185]:


total_features = []
train_labels = []
for i in range(train_reader.get_number_of_examples()):
    all_features, label = extract_features_from_reader(train_reader, i,'train')
    total_features.append(all_features)
    train_labels.append(label)


# In[208]:


from sklearn.preprocessing import StandardScaler
import numpy as np
X = np.array(total_features)
# Define the indices of the categorical features
categorical_indices = list(range(495, 511)) + list(range(512, 518))

# Define the indices of the numerical features
numerical_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]

# Extract the numerical features
X_numerical = X[:, numerical_indices]

# Normalize the numerical features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X_numerical)

# Put the normalized features back into the original array
X[:, numerical_indices] = X_numerical


# In[211]:


X[0]


# In[194]:


labels_2d = np.array(train_labels)
labels_1d = labels_2d.reshape(-1)


# In[197]:


labels_1d


# In[184]:


train_reader.get_number_of_examples()


# In[189]:


example = train_reader.read_example(1)


# In[191]:


lis = []

lis.append(example['y'])


# In[192]:


lis


# In[ ]:




