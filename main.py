import pandas as pd  # load and manipulate data and for One-Hot encoding
import numpy as np  # data manipulation
import matplotlib.pyplot as plt  # drawing graphs
import matplotlib.colors as colours
from sklearn.utils import resample  # downsample the dataset
from sklearn.model_selection import train_test_split  # split data to training and testing set
from sklearn.preprocessing import scale  # scale and centre data
from sklearn.svm import SVC  # will make a Sv classifier
from sklearn.model_selection import GridSearchCV  # will do cross validation
from sklearn.metrics import confusion_matrix  # this creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix  # draws a confusion matrix
from sklearn.decomposition import PCA  # perform PCA to plot the data

# ---- Importing data ----
df = pd.read_csv('default of credit card clients.tsv',
                 header=1,  # NOTE: second line contains column names, 1 line is skipped as header
                 sep='\t')  # NOTE: Pandas auto detects delimeters, just make sure we use seperator = tab

# for direct dataset use
# df = pd.read_excel('linkName', header=1, sep='\t)
# print(df.head())  # prints 5 lines of dataset


# ---- Formating Data ----
df.rename({'default payment next month': 'DEFAULT'}, axis='columns', inplace=True)
# print(df.head())  # Changed last column name to default

df.drop('ID', axis=1, inplace=True)
# print(df.head()) # Removed ID column

# print(df.dtypes)  # check data type is same and no string N/A for missing

# print(df['SEX'].unique())  # check if it contains 1/2
# print(df['EDUCATION'].unique()) # check if it contains 1,2,3,4 only
# print(df['MARRIAGE'].unique()) # check if it contains 1,2,3 only


# ---- Assume 0 is missing data ----

# print(len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)]))  # 68 rows have it
# print(len(df))  # out of 30000

# As 68 is barely 1% of 30000, we will remove them from data frame

# store the rows with no missing data
df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]

# print(len(df_no_missing))  # should have 30,000 - 68 = 29932 rows

# ---- Check if they contain wrong values anymore ----
# print(df_no_missing['EDUCATION'].unique())
# print(df_no_missing['MARRIAGE'].unique())


# ---- Downsampling ---
df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]

df_no_default_downsampled = resample(df_no_default,
                                     replace=False,
                                     n_samples=1000,
                                     random_state=42)

# print(len(df_no_default_downsampled))

df_default_downsampled = resample(df_default,
                                  replace=False,
                                  n_samples=1000,
                                  random_state=42)

# print(df_no_default_downsampled)
# print(df_default_downsampled)


# concatenate the two data lists
df_downsampled = pd.concat([df_no_default_downsampled, df_default_downsampled])
# print(df_downsampled)


# ---- Format Data ----

# Split into Independent(X) and Dependent(y) Variables
# Independent - what we change
# Dependent - What we want to predict (dependent on Independent var)

# drop DEFAULT, everything except what we want to predict
X = df_downsampled.drop('DEFAULT', axis=1).copy()  # alt: X = df_no_missing.iloc[:,:-1].copy()
# print(X.head())

# only what we want to predict
y = df_downsampled['DEFAULT'].copy()
# print(y.head())


# ---- One-Hot Encoding ----
# One hot encoding is a process by which categorical variables are
# converted into a form that could be provided to
# ML algorithms to do a better job in prediction

# sci-kit doesn't support categorical data

X_Encoded = pd.get_dummies(X, columns=['SEX',
                                       'EDUCATION',
                                       'MARRIAGE',
                                       'PAY_0',
                                       'PAY_2',
                                       'PAY_3',
                                       'PAY_4',
                                       'PAY_5',
                                       'PAY_6'])

# print(X_Encoded.head())


# ---- Center and Scale Data ----
# Data needs: mean value == 0  &  standard deviation = 1

# Radical Basis Function (RBF) assumes data is centred and scaled
# Data split to training and testing datasets
# and scaled separately to prevent influence on each other
# Or, Data Leakage
# y_train and y_test are 0s and 1s so no scaling needed.

X_train, X_test, y_train, y_test = train_test_split(X_Encoded, y, random_state=42)

# print(y_train)

X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# print(np.unique(y_train))

# ---- Building a Preliminary SVM ----

# clf_svm = SVC(random_state=42)
# clf_svm.fit(X_train_scaled, y_train)

# plot_confusion_matrix(clf_svm,
#                      X_test_scaled,
#                      y_test,
#                      values_format='d',
#                      display_labels=["Did not default", "Defaulted"])

# plt.show()


# ---- Optimizing parameters using CV ----
# parameters we will test for cross validation
# default values included as they are possible answers

'''
param_grid = [
    {
        'C': [.5, 1, 10, 100],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf'],
    }
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    # scoring = 'balanced_accuracy',  # slightly improved, but hardly C=1, gamma=0.01
    # scoring = 'f1',  # Terrible! C=0.5, gamma=1
    # scoring = 'f1_micro',  # Slightly improved, but hardly c=1, gamma=0.01
    # scoring = 'f1_macro',  # Same, C = 1, gamma='scale'
    # scoring = 'f1_weighted', # Same, C = 1, gamma = 'scale'
    # scoring = 'roc_auc', # Terrible, C = 1, gamma = 0.001
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=0  # to see what Grid Search is doing, set it to 2
)

optimal_params.fit(X_train_scaled, y_train)
# print(optimal_params.best_params_)
'''

# ---- Building and Drawing Final SVM ----

clf_svm = SVC(random_state=42, C=100, gamma=0.001)
clf_svm.fit(X_train_scaled, y_train)

plot_confusion_matrix(clf_svm,
                      X_test_scaled,
                      y_test,
                      values_format='d',
                      display_labels=["Did not default", "Defaulted"])

# plt.show()

# Number of columns
# print(len(df_downsampled.columns))


# -- Using Principal Component Analysis to shrink 24D to 2D and plot scree plot --

pca = PCA()  # By default, only center the data
X_train_pca = pca.fit_transform(X_train_scaled)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var) + 1)]

plt.bar(x=range(1, len(per_var) + 1), height=per_var)

plt.tick_params(
    axis='x',  # changes apply to x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along bottom edge are off
    top=False,  # ticks along upper edge are off
    labelbottom=False  # labels along the bottom edge are off
)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
# plt.show()
