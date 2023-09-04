#!/usr/bin/env python
# coding: utf-8

# In[26]:


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


# In[5]:


import os
import numpy as np
import random
import pandas as pd

class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


# In[6]:


class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


# In[7]:


def get_los(data, root_dir, mode='train'):
    # Extract the patient ID and episode number from the 'name' field
    name_parts = data['name'].split('_')
    patient_id = name_parts[0]
    episode_number = name_parts[1]

    # Construct the path to the patient folder
    patient_folder = os.path.join(root_dir, mode, patient_id)

    # Load the episode data
    episode_path = os.path.join(patient_folder, f'{episode_number}.csv')
    episode_data = pd.read_csv(episode_path)

    # Get the ICU stay ID from the episode data
    icu_stay_id = episode_data['Icustay'].values[0]

    # Load the stays data
    stays_path = os.path.join(patient_folder, 'stays.csv')
    stays_data = pd.read_csv(stays_path)

    # Get the length of stay for the specific ICU stay ID
    los = stays_data[stays_data['ICUSTAY_ID'] == int(icu_stay_id)]['LOS'].values[0]

    return los


# In[8]:


dataset_dir = "C:\\Users\\samit\\OneDrive\\Desktop\\Datasets\\data\\in-hospital-mortality"
root_dir = r'C:\Users\samit\OneDrive\Desktop\Datasets\data\root'


# In[9]:


train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(dataset_dir, 'train'),
                                         listfile=os.path.join(dataset_dir, 'train_listfile.csv'),
                                         period_length=48.0)


val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(dataset_dir, 'train'),
                                       listfile=os.path.join(dataset_dir, 'val_listfile.csv'),
                                       period_length=48.0)

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(dataset_dir, 'test'),
                                       listfile=os.path.join(dataset_dir, 'test_listfile.csv'),
                                       period_length=48.0)


# In[10]:


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


# In[17]:


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
    labels = get_los(example_data,root_dir,root)
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
    
    
    return all_features, labels 
# Example usage
# all_features = extract_features_from_reader(train_reader, 0)


# In[18]:


train_features = []
train_labels = []
for i in range(train_reader.get_number_of_examples()):
    all_features, label = extract_features_from_reader(train_reader, i,'train')
    train_features.append(all_features)
    train_labels.append(label)


# In[19]:


val_features = []
val_labels = []
for i in range(val_reader.get_number_of_examples()):
    all_features, label = extract_features_from_reader(val_reader, i,'train')
    val_features.append(all_features)
    val_labels.append(label)


# In[20]:


test_features = []
test_labels = []
for i in range(test_reader.get_number_of_examples()):
    all_features, label = extract_features_from_reader(test_reader, i,'test')
    test_features.append(all_features)
    test_labels.append(label)


# In[21]:


train_features = np.array(train_features)
val_features = np.array(val_features)
test_features = np.array(test_features)


# In[22]:


train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)


# In[24]:





# In[ ]:





# In[27]:


# Define the indices of the categorical features
categorical_indices = list(range(495, 511)) + list(range(512, 518))

# Define the indices of the numerical features
numerical_indices = [i for i in range(train_features.shape[1]) if i not in categorical_indices]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both
train_features[:, numerical_indices] = scaler.fit_transform(train_features[:, numerical_indices])

# Use the same scaler to transform the validation and test data
val_features[:, numerical_indices] = scaler.transform(val_features[:, numerical_indices])
test_features[:, numerical_indices] = scaler.transform(test_features[:, numerical_indices])


# In[29]:


train_features[0]


# In[32]:


test_labels


# In[ ]:





# In[24]:





# In[ ]:





# In[33]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(train_features)
X_val = imputer.transform(val_features)
X_test = imputer.transform(test_features)


# In[34]:


np.save('X_train_l.npy',X_train)
np.save('X_val_l.npy',X_val)
np.save('X_test_l.npy',X_test)
np.save('train_labels_l',train_labels)
np.save('val_labels_l',val_labels)
np.save('test_labels_l',test_labels)


# In[35]:


import pandas as pd

# Define the bin edges
bins = [2, 3, 4, 5, 6, 7, 8, 14, float('inf')]

# Define the bin labels
labels = ['2-3 days', '3-4 days', '4-5 days', '5-6 days', '6-7 days', '7-8 days', '8-14 days', '14+ days']

# Map the train_labels to the bins
train_labels_binned = pd.cut(train_labels, bins=bins, labels=labels, right=False)
val_labels_binned = pd.cut(val_labels, bins=bins, labels=labels, right=False)
test_labels_binned = pd.cut(test_labels, bins=bins, labels=labels, right=False)
# One-hot encode the binned labels
train_labels_onehot = pd.get_dummies(train_labels_binned)
val_labels_onehot = pd.get_dummies(val_labels_binned)
test_labels_onehot = pd.get_dummies(test_labels_binned)
print("One-hot encoded train labels:")
train_labels_onehot.head()


# In[37]:


train_label = train_labels_onehot.values
val_label = val_labels_onehot.values
test_label= test_labels_onehot.values


# In[42]:


train_label[1:3]


# In[43]:


X_train[0]


# In[36]:


import numpy as np

# Assume your_array is the array you want to check
num_nan = np.isnan(X_train).sum()

print(f"Total number of NaN values in the array: {num_nan}")


# In[102]:


from sklearn.linear_model import LogisticRegression
y_train_label_encoded = np.argmax(train_label, axis=1)
y_test_label_encoded = np.argmax(test_label, axis=1)
# Initialize the Logistic Regression model with One-vs-Rest strategy
model = LogisticRegression(penalty = 'l2',C = 0.001,multi_class='multinomial', solver='lbfgs', max_iter=10000)

# Fit the model
model.fit(X_train, y_train_label_encoded)


# In[103]:


# Get predicted probabilities
y_pred_proba = model.predict_proba(X_test)

# Convert predicted probabilities to label-encoded form
y_test_pred = np.argmax(y_pred_proba, axis=1)


# In[104]:


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize



# Compute the micro and macro ROC AUC score
roc_auc_micro = roc_auc_score(test_label, y_pred_proba, multi_class='ovr', average='micro')
roc_auc_macro = roc_auc_score(test_label, y_pred_proba, multi_class='ovr', average='macro')

print(f"Micro AUC-ROC: {roc_auc_micro}")
print(f"Macro AUC-ROC: {roc_auc_macro}")


# In[78]:


import seaborn as sns
# Model names and AUC-ROC values
model_names = ["Logistic Regression","Normal LSTM", "Bidirectional LSTM", "LSTM without Multimodality", "Channel-wise LSTM"]
micro_auc_values = [0.7595689643505967,0.782905169188585, 0.7740942004864486, 0.7867237074109105, 0.7782539305408189]
macro_auc_values = [0.622014295026788,0.65,0.64,0.66,0.64]
# Combine the data and sort by Micro AUC-ROC values
data = list(zip(model_names, micro_auc_values, macro_auc_values))
data.sort(key=lambda x: x[1], reverse=True)  # Sort by Micro AUC-ROC
sorted_model_names, sorted_micro_auc, sorted_macro_auc = zip(*data)

# Set up the bar positions
bar_width = 0.35
index = np.arange(len(sorted_model_names))

# Create the bars
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Bars for Micro AUC-ROC
bar1 = plt.bar(index - bar_width/2, sorted_micro_auc, bar_width, color=sns.color_palette("pastel")[0], label='Micro AUC-ROC')

# Bars for Macro AUC-ROC
bar2 = plt.bar(index + bar_width/2, sorted_macro_auc, bar_width, color=sns.color_palette("pastel")[1], label='Macro AUC-ROC')

# Overlaying line chart
plt.plot(index, sorted_micro_auc, color=sns.color_palette("dark")[0], marker='o', label='Micro AUC-ROC Trend')
plt.plot(index, sorted_macro_auc, color=sns.color_palette("dark")[1], marker='o', label='Macro AUC-ROC Trend')

# Label the chart
plt.xlabel('Model', fontsize=14)
plt.ylabel('AUC-ROC Value', fontsize=14)
plt.title('Comparison of AUC-ROC values across models', fontsize=16)
plt.xticks(index, sorted_model_names, rotation=45, ha='right', fontsize=12)
plt.ylim(0.6, 0.8)
plt.legend()

# Display the chart
plt.tight_layout()
plt.show()


# In[79]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model names and AUC-ROC values
model_names = ["Logistic Regression","Normal LSTM", "Bidirectional LSTM", "LSTM without Multimodality", "Channel-wise LSTM"]
micro_auc_values = [0.7595689643505967,0.782905169188585, 0.7740942004864486, 0.7867237074109105, 0.7782539305408189]
macro_auc_values = [0.622014295026788,0.65,0.64,0.66,0.64]

# Set up the bar positions
bar_width = 0.6
index = np.arange(len(model_names))

# Create the bars
sns.set_style("whitegrid")

# Plot for Micro AUC-ROC
plt.figure(figsize=(12, 6))
bar1 = plt.bar(index, micro_auc_values, bar_width, color=sns.color_palette("pastel")[0], label='Micro AUC-ROC')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Micro AUC-ROC Value', fontsize=14)
plt.title('Comparison of Micro AUC-ROC values across models', fontsize=16)
plt.xticks(index, model_names, rotation=45, ha='right', fontsize=12)
plt.ylim(0.6, 0.8)
for i, v in enumerate(micro_auc_values):
    plt.text(i, v + 0.005, "{:.2f}".format(v), ha='center')
plt.tight_layout()
plt.show()

# Plot for Macro AUC-ROC
plt.figure(figsize=(12, 6))
bar2 = plt.bar(index, macro_auc_values, bar_width, color=sns.color_palette("pastel")[1], label='Macro AUC-ROC')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Macro AUC-ROC Value', fontsize=14)
plt.title('Comparison of Macro AUC-ROC values across models', fontsize=16)
plt.xticks(index, model_names, rotation=45, ha='right', fontsize=12)
plt.ylim(0.6, 0.8)
for i, v in enumerate(macro_auc_values):
    plt.text(i, v + 0.005, "{:.2f}".format(v), ha='center')
plt.tight_layout()
plt.show()


# In[106]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model names and AUC-ROC values
model_names = ["Logistic Regression","Normal LSTM", "Bidirectional LSTM",  "Channel-wise LSTM","LSTM without Multimodality"]
micro_auc_values = [0.7635689643505967,0.782905169188585, 0.7740942004864486, 0.7867237074109105, 0.7782539305408189]
macro_auc_values = [0.624014295026788,0.6494474704406468,0.64,0.66,0.64]

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
plt.ylabel('Micro AUC-ROC Value', fontsize=14)
plt.title('Comparison of Micro AUC-ROC values across models', fontsize=16)
plt.xticks(index, sorted_model_names_micro, rotation=45, ha='right', fontsize=12)
plt.ylim(0.72, 0.8)
plt.plot(index, sorted_micro_auc, color='red', marker='o', linestyle='dashed', linewidth=1, markersize=5)
for i, v in enumerate(sorted_micro_auc):
    plt.text(i, v + 0.005, "{:.2f}".format(v), ha='center')
plt.tight_layout()
plt.show()

# Plot for Macro AUC-ROC
plt.figure(figsize=(12, 6))
bar2 = plt.bar(index, sorted_macro_auc, bar_width,  color=sns.color_palette("pastel")[1], label='Macro AUC-ROC')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Macro AUC-ROC Value', fontsize=14)
plt.title('Comparison of Macro AUC-ROC values across models', fontsize=16)
plt.xticks(index, sorted_model_names_macro, rotation=45, ha='right', fontsize=12)
plt.ylim(0.6, 0.68)
plt.plot(index, sorted_macro_auc, color='red', marker='o', linestyle='dashed', linewidth=1, markersize=5)
for i, v in enumerate(sorted_macro_auc):
    plt.text(i, v + 0.005, "{:.2f}".format(v), ha='center')
plt.tight_layout()
plt.show()


# In[76]:


# Plot the AUC-ROC curves for each model in a single plot
plt.figure(figsize=(8, 6))
plot_roc_curve(model, X_test, test_labels, 'Normal LSTM', 'blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[77]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2', 'elasticnet', 'none'],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#     'max_iter': [50, 100, 200],
#     'class_weight': ['balanced', None]
# }

# # Create a logistic regression model
# logistic = LogisticRegression()

# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=logistic, param_grid=param_grid, 
#                            cv=5, n_jobs=-1, verbose=2)

# # Fit the grid search to the data
# grid_search.fit(X_train, train_labels)

# # Get the best parameters
# best_params = grid_search.best_params_

# # Create a new model with the best parameters
# best_model = LogisticRegression(**best_params)

# # Fit the best model
# best_model.fit(X_train, train_labels)

# print(f"Best Parameters: {best_params}")


# In[ ]:





# In[48]:


rf = RandomForestClassifier()
rf.fit(X_train, train_labels)


# In[49]:


y_pred_r = rf.predict(X_test)


# In[50]:


test_roc_auc_rf = roc_auc_score(test_labels, y_pred_r)


# In[51]:


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




