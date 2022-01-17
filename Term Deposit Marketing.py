#pip install missingno
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def load_application_train():
    data = pd.read_csv("hangikredi.com/term-deposit-marketing-2020.csv")
    return data

df = load_application_train()
df.head()

#First, we observe the dataset in a general framework.
def check_df(dataframe, head=5, tail = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head ######################")
    print(dataframe.head(head))
    print("##################### Tail ######################")
    print(dataframe.tail(tail))
    print("##################### NA ########################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df["y"].replace("yes", 1, inplace=True)
df["y"].replace("no", 0, inplace=True)
df.head()
df.info()

df["y"] = df["y"].astype(float)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.
    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                Class threshold for numeric but categorical variables
        car_th: int, optinal
                Class threshold for categorical but cardinal variables
    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of 3 lists with return is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#The num_but_cat variable is actually inside the cat_cols variable. Printed for reporting purposes.
cat_cols

num_cols

df[num_cols].describe().T

df[cat_cols].describe()

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)

outlier_thresholds(df, cat_cols)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, num_cols)

for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, num_cols)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in df.columns:
    target_summary_with_num(df, "y", col)

cor = df.corr(method="pearson")
cor

sns.heatmap(cor)
plt.show()
#After performing correlation analysis, we perform LOF (Local Outlier Factor) analysis.

def assing_missing_values(dataframe, except_cols):
    for col in dataframe.columns:
        dataframe[col] = [val if val!=0 or col in except_cols else np.nan for val in df[col].values]
    return dataframe

df = assing_missing_values(df, except_cols=["duration", "y"])

df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_cols = missing_values_table(df, True)
#The "msno.matrix()" method is a tool that shows whether the deficiencies in the observations of the variables come together or not.
#If deficiencies in the variables occur together, there will be deficiencies in other variables as well.
msno.bar(df)
plt.show()

msno.heatmap(df)
plt.show()

#The "msno.heatmap()" method displays a heatmap based on shortcomings
#Here we check if the missing values come out with a certain correlation, that is, we are interested in its randomness.
#Co-occurrence of deficiencies or deficiencies dependent on a particular variable are both dependency scenarios.


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_Na_Flag'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_Na_")].columns

    for col in na_flags:
        print(pd.DataFrame({"Target_Mean": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "y", na_cols)
#Here we compare the "y" variable with the missing variables (na_cols).
#We use it in the "missing_vs_target" function to see how missing values affect the target variable.



df = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

#We use the "get_dummies" method to do one hot encoding and label encoding at the same time, and if we set its "drop_first" argument to True, categorical variables with two classes will discard the first class and keep the second class.
#So we represent this categorical variable in a binary way.

df.head()



scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()

##Since we need to standardize the variables, we call the "MinMaxScaler" method and assign it to the scaler. The "MinMaxScaler" method compresses values between 0 and 1 min-max.
# Then we apply the "scaler" to our dataset with the fit_transform method. Since the converted data will not be in the format we want, we convert it to a data set using the pd.
# DataFrame method and get it from dff.columns.

imputer = KNNImputer()
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df.head()

#By creating the model object, we define the number of neighborhoods as 5. This method works like this, to put it simply;
#It selects the missing observation unit in the variable with missing value in our data set,
#takes the average of the ages of the 5 closest neighbors of this observation, and assigns the missing observation.


df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
df.head()
#Here we look at the normalized form of missing values that we filled in by recycling the data set we standardized earlier.



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

df.isnull().sum()


cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

def one_hot_encoder(dataframe, categorical_columns, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

#When we sent the dependent variable and categorical variables to this function,
#it became a function that would bring the operations we have done above.
#When we sent the dependent variable and categorical variables to this function, it became a function that would bring the operations we have done above.
#First, we do the information about how many classes the relevant categorical variable has, second the class frequencies,
#the third the class frequency ratios, and fourthly, the groupby process according to the dependent variable.



def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe

#Here we have brought together all classes below 0.01 and called these classes rare classes.
# We created a function called "rare_encoder", since changes will be made in this dataframe,
# we created a name called temp_df and copied it here. At the beginning of the function,
# we defined an argument named "rare_perc" and we said that if there is a class ratio of any categorical variable
# in rare_columns that is lower than the value entered in the "rare_perc" argument, and at the same time,
# if it is a categorical variable, we bring them as "rare_columns".
# Then we navigate through these rare_columns and get their "value_counts( )" and divide by the total number of observations
# to calculate the class ratios for a variable in the "temp_df" dataset.
# After reducing the relevant variable to the number of the "rare_perc" argument we entered at the beginning,
#we keep the remaining indexes by creating a place called "rare_labels".
# Then, if we observe any "rare_labels" in the "rare_columns" that we navigate in "temp_df", we print "Rare" there, otherwise we leave it as normal.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "y", cat_cols)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

ohe_cols

df = one_hot_encoder(df, ohe_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
useless_cols

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.head()

y = df["y"]
X = df.drop(["y"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=47)

rf_model = RandomForestClassifier(random_state=2).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

print("Accuracy Score: " + f'{accuracy_score(y_pred, y_test):.2f}')

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)


# create dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# prepare the cross-validation procedure
cv = KFold(n_splits=5, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))











