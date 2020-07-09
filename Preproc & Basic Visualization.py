### TO ADD: Mean encoding, value encoding methods

######################### Working directory #########################
import os
os.getcwd()
os.chdir("C:\\Users\\lzhga\\Documents\\")

######################### Importing Datasets #########################
df = pd.read_excel('default of credit card clients.xls')
df = pd.read_excel(xls, 'Sheet1')
df = pd.read_csv("nba.csv", keep_default_na=False, na_values=[""])) 
df = pd.read_excel (r'C:\Users\Ron\Desktop\Product List.xlsx')
df = pd.read_clipboard()

## Only some columns
data = pd.read_excel (r'C:\Users\Ron\Desktop\Product List.xlsx') 
df = pd.DataFrame(data, columns= ['Product','Price'])

df = pd.read_csv("nba.csv", usecols=["Name"], header=2, skiprows=[3, 4] # Header is the 2nd column (Starts from 0), skips lines 3 and 4, only read Name column.

######################### Understanding more about dataset #########################
df.info
df.nunique() # Shows number of unique values
df.head(5)
df.isnull().any() # Does not have any missing values
df.shape #  Returns dimensions of df
                 
df[columm].describe() # Shows min max etc. 
df[column].unique() # Returns array of unique values
df[column].value_counts() # Count number of unique values
                
# Function: Return no. of missing values for all columns
def num_missing(x):
  return sum(x.isnull())
print ("Missing values per column:")
print (df.apply(num_missing, axis=0)) 

# Function: Print value_counts for all or specific columns
def countUV(x, df):
  for i in df[x].columns:
    print(i)
    print(df[i].value_counts())
    print("\n")  
countUV(df.columns.values, df)
                 
######################### Manipulation #########################

## Datetime values ##
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["day"] = df["Datetime"].dt.day
df['month'] = df["Datetime"].dt.month
df['year'] = df["Datetime"].dt.year
df['Day of Week'] = df['Datetime'].apply(lambda time: time.dayofweek) # 0 = Monday, 6 = Sunday
df1 = df['Day'].map(str) + '-' + df['Month'].map(str) + '-' + df['Year'].map(str) # Concatenating back
df["Datetime"] = pd.to_datetime('20190108',format='%Y%d%m'). 
df["Datetime"].dt.month_name() # Returns month name, can use days_in_month(), is_month_end etc.
df[df["Datetime"].dt.is_month_end] # Returns month end results

## Changing type to numeric/int/category
df[col] = pd.to_numeric(df[col], errors="coerce") # Change those errors into NaNs
df = df.astype(int)
df[col]=df[col].astype('int')
df[col] = df[col].astype('category')

## Function: Splitting dataframe into categorical and continuous
def cat(df):
  categorical_columns = []
  for c in df.columns:
      if df[c].map(type).eq(str).any(): #check if there are any strings in column
          categorical_columns.append(c)
      else:
        next
  df_cat = df.select_dtypes(include=['object']).copy() # Only select data type = object
  return df_cat
df_cat = cat(df)
  
def cont(df):
  numeric_columns = []
  for c in df.columns:
      if df[c].map(type).eq(str).any(): #check if there are any strings in column
        next
      else:
        numeric_columns.append(c)
  df_cont = df.select_dtypes(include=['float','int64']).copy() # Only select data type = float/int64
  return df_cont
df_cont = cont(df)
 
# Concatenate Dataframes back
df_combined = pd.concat([df_cat, df_cont], axis = 1)
                 
## Sorting
df.sort_values(
  by=['Country name','Year'], 
  ascending=[False,True]
data.sort_index() # Sort by index

## Renaming variables
df.rename(columns={'default payment next month':'y'}, inplace=True)
# Renaming many variables
new_names =  {'Unnamed: 0': 'Country',
              '? Summer': 'Summer Olympics',
              '01 !': 'Gold',
              '02 !': 'Silver'}
df.rename(columns=new_names, inplace = True)

## Reordering labels for plotting
df["month"] = pd.Categorical(df["month"],["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], ordered=True) # Ordering categorical values

## Remapping cell values
df["y"] = df["y"].map({'no': 0, 'yes': 100}) ### Mapping no to 0 and yes to 100, so that average produces y percent
df["BinnedAge"] = df["BinnedAge"].map({ 29 : "Younger Than 30", 31 : "30+"})
# Remapping cell values - Using dictionary
cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
df.replace(cleanup_nums, inplace=True)
  
# Remapping respective cell in c2 to 10, based on condition c1 == Value
df.loc[df['c1'] == 'Value', 'c2'] = 10

## Function - Discretizing (Or use k-means clustering)
def cleancols(column):
    for i in column:
       for index, row in df.iterrows():
          if df.at[index, i] > 3:
               df.at[index, i] = 4
          elif df.at[index, i] == 0:
              df.at[index, i] = 4            

# Function - Discretizing v2 (Or use k-means clustering)
def disc(num):
    if num <5000000:
        return "Low"
    elif num >= 5000000:
        return "High"
df["SalaryCat"] = df["Salary"].apply(disc)
  
## Using iloc to select rows
df.iloc[0:3] # Returns rows with index values 0, 1, 2
df = df.iloc[1:] # Select row 2 onwards
df.iloc[2:, -3:] # Returns from row 3 onwards, columns 3rd last to last
df.iloc[(df['Age'] < 30).values, [1, 3]] # Can only use integers for column reference

## Using loc to select specific rows based on condition(s)
# https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
df.set_index("last_name", inplace=True) # Setting column ["last_name"] as index
name = ["Andreas", "Veness"]
df.loc[name] # Returns df with index values "Andreas" and "Veness"
  
df.loc[df["City"] == "Singapore", ["email", "number"]] # Returns dataframe with columns email and number, which was indexed by last name, with City column = Singapore
df.loc[(df["Age"] < 20) & (df["Number"] == 1.0)] # Multiple Conditions

## Using at - Used to access cells of a specific column
df.at[2, "Name"] # Returns row 2 of column "Name"
# If both yes, return yes
has_loan_list = []
for index, row in df.iterrows():
    if df.at[index, "housing"] == "yes" or df.at[index, "loan"] == "yes":
        has_loan_list.append("yes")
    else:
        has_loan_list.append("no")
df["has_loan"] = has_loan_list

## Dropping rows (axis=0) / columns (axis=1)
df = df.drop([col], axis=1)
df = df[df["Name"] != 'Tina'] # Delete rows where Name = Tina
df = df.drop(0) # Drop index row = 0
df.dropna() # Drops rows
df.dropna(axis='columns') # Drops columns
df.dropna(how='all') # Drops rows where ALL row cells are na
df.dropna(thresh=2) # Drops rows based on threshold
  
to_drop = ['Edition Statement','Corporate Author']
df.drop(to_drop, inplace=True, axis=1)

## Dropping rows based on one column NaN
df = df[pd.notnull(df[col])]
df[df[col].notnull()]

# Delete the rows with label "Ireland"
# For label-based deletion, set the index first on the dataframe:
data = data.set_index("Area")
data = data.drop("Ireland", axis=0). # Delete all rows with label "Ireland"
  
## Adding/Append row using dict
df.append({"index":"Test", "Expenses":99999, "Revenue":99999, "SubDept":"C", "Cost Margin":9}, ignore_index=True)
  
# Adding/Append rows using pd.Series
listOfSeries = [pd.Series(['Raju', 21, 'Bangalore', 'India'], index=df.columns ) ,
                pd.Series(['Sam', 22, 'Tokyo', 'Japan'], index=df.columns ) ,
                pd.Series(['Rocky', 23, 'Las Vegas', 'US'], index=df.columns ) ]
upd_df = df.append(listOfSeries , ignore_index=True)
  
# Adding/Append from another df
result = df1.append(df4, sort=False) #Default = axis=0, which means df4 below df1, appended by similar column labels

## Concatenate Dataframes, Merging Dataframes
data_joined = pd.concat([obj_df, float_df], axis = 1)

## Merge / Concatenate
#https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
df_merged = pd.merge(df1, df2, how='left', on="Order ID") # Similar to left join in SQL, default = innerjoin. Merge based on similar columns "Order ID"
df_inner = pd.merge(df1, df2, left_on="Order ID", right_on="New ID") # Inner join based on Order ID = New ID
df_concat = pd.concat([df_grouped, df2_grouped], axis=1, sort=False)
  
# Adding columns as a function of another column
df.assign(Revised_Salary = lambda x: df['Salary'] 
                               + df['Salary']/10) 
df["Age+3"] = df["Age"].apply(lambda x: x+3)
 
# Adding columns based on condition of multiple columns
def fun_eq(row):
    if row["A"] == row["B"]:
        val = 0
    elif row["A"] != row["B"]:
        val = 1
    else:
        val = -1
    return val
df["AB"] = df.apply(fun_eq, axis=1)

## Imputing missing value numeric data
## Filling based on specific conditions https://stackoverflow.com/questions/49088259/how-to-impute-values-in-a-column-when-certain-conditions-are-fulfilled-in-other
ValIndex = ["Age"]
for col in ValIndex: #ValIndex has the columns name
    df[col] = df[col].fillna(df[col].mean())

# Fill NA with mean of group
df['Salary'] = df['Salary'].fillna(df.groupby('Team')['Salary'].transform('mean'))

# Fill NA with most frequent value
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

## Imputing missing categorical data
obj_df[obj_df.isnull().any(axis=1)] # Returning rows with nulls in any column
obj_df["num_doors"].value_counts()
obj_df = obj_df.fillna({"num_doors": "four"}) # Fill NaNs with "four"

## Imputing missing values for train/test
import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_train)
print(imp.transform(X_train))   # Using properties of train data to use on train data   
print(imp.transform(X_test))   # Using properties of train data to use on test data   

#### Encoding ####

## Label Encoding ##
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass
# Alternatively, label encode one by one
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['contact']     = le.fit_transform(df['contact']) 
df['month']       = le.fit_transform(df['month']) 

## One Hot Encoding ##
category_col = ["A", "B", "C"]
for i in category_col:
    df = df.join(pd.get_dummies(raw[i], prefix=i))
    df = df.drop([i], axis=1)

######################### PLOTTING #########################
# Function: Plotting Categorical Variables
def plotUniqueVal(dataframe, column):
    sns.set(style="darkgrid")
    ax = sns.catplot(x=column, data=dataframe, kind='count', aspect=0.6)
    ax.set_xticklabels(rotation=90)
    ax.fig.suptitle("Groups and frequency in attribute: {}".format(column))
#         ax.savefig("{}Freq.png".format(name))
    return
plotUniqueVal(df, "y")
  
## Plotting Categorical Variables - with exceptions for some columns
def plotUniqueVal(dataframe, column):
    sns.set(style="darkgrid")
    monthNames = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    for name in column:
        if name == "month": 
            ax = sns.catplot(x=name, data=dataframe, kind='count', order=monthNames)
        elif name == "job": 
            ax = sns.catplot(x=name, data=dataframe, kind='count', )
            ax.set_xticklabels(rotation=90)
        else:
            ax = sns.catplot(x=name, data=dataframe, kind='count', aspect=0.6)
            ax.set_xticklabels(rotation=90)
        ax.fig.suptitle("Groups and frequency in attribute: {}".format(name))
#         ax.savefig("{}Freq.png".format(name))
    return
plotUniqueVal(df, "y")

# Function: Plotting continuous variable, notnull()
def plotDist(dataframe, column):
    sns.set(style="darkgrid")
    ax = sns.distplot(dataframe[dataframe[column].notnull()][column], norm_hist=True)
plotDist(df, "y")

# Binned - Age vs. Average Salary
Binned_df = df.copy()
Binned_df["Age"] = pd.cut(Binned_df["Age"], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90], labels=["10-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90"])
attr_avg = Binned_df.groupby("Age", as_index=False, sort=True).mean()
attr_count = Binned_df.groupby("Age", as_index=False, sort=True).count()
sns.set(style="darkgrid")
ax = sns.barplot(data = attr_count, x = "Age", y = "Salary")
for item in ax.get_xticklabels():
  item.set_rotation(45)
# ax2 = plt.twinx()
# ax2 = sns.lineplot(data = attr_avg, x = "Age", y = "Salary", ax=ax2, sort=False, color="black")
# ax2.set_ylabel("Avg. Salary (Line)")
ax.set_ylabel("Count (Bar)")
ax.set_title("Attribute: {}".format("Age"))
fig = ax.get_figure()
# fig.savefig("{}_CountBalance.png".format("age"), bbox_inches="tight")

## Correlation Plot
continuousRaw2 = ["age", "balance", "duration", "campaign"]
def plotpairs(dataframe, column):
    ax = sns.pairplot(data=dataframe[column], dropna=True)
#     ax.savefig("pairsplot.png")
    return
plotpairs(df, continuousRaw2)

## Combined yPct & count vs Continuous Binned Variable ##
def plotConCountYPct(attr, bins, labels):
    Binned_df = df.copy()
    Binned_df[attr] = pd.cut(Binned_df[attr], bins=bins, labels=labels)
    attr_avg = Binned_df.groupby(attr, as_index=False, sort=True).mean()
    attr_count = Binned_df.groupby(attr, as_index=False, sort=True).count()
    sns.set(style="darkgrid")
    ax = sns.barplot(data = attr_count, x = attr, y = "y")
    ax2 = plt.twinx()
    ax2 = sns.lineplot(data = attr_avg, x = attr, y = "y", ax=ax2, sort=False, color="black")
    ax.set_ylabel("Count (Bar)")
    ax2.set_ylabel("% success (Line)")
    ax2.set_ylim(0,100)
    ax.set_title("Attribute: {}".format(attr))
    fig = ax.get_figure()
    fig.savefig("{}_CountYPct.png".format(attr), bbox_inches="tight")
plotConCountYPct("age", [10, 20, 30, 40, 50, 60, 70, 80, 90], ["10-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90"])

## Scatterplots
def plotScatter(dataframe, x, y, hue):
    ax = sns.scatterplot(x=x, y=y, hue=hue, data=dataframe)
    fig = ax.get_figure()
    fig.savefig("{}_{}_scatter.png".format(x, y), bbox_inches="tight")

# plotScatter(df, "age", "campaign", "yCat")
# plotScatter(df, "age", "balance", "yCat")
# plotScatter(df, "campaign", "duration", "yCat")

## Unstacked Plots - pdays vs poutcome, % success ##
ax = df.groupby(["pdays", "poutcome"], as_index=True, sort=True).mean()["y"].unstack().plot()
ax.set_ylabel("% success")
ax.set_title("pdays by poutcome")
fig = ax.get_figure()
fig.savefig("pdays by poutcome.png")

## Line plots
sns.lineplot(x="Age", y="Salary", hue="y", data=df)

## Binning then plotting - 2 Separate graphs
for index, row in df.iterrows():
    if df.at[index, "Age"] < 30 or df.at[index,"Age"] > 35:
        df.at[index, "BinnedAge"] = 29
    elif df.at[index, "Age"] > 29:
        df.at[index, "BinnedAge"] = 31
df["BinnedAge"] = df["BinnedAge"].map({ 29 : "Younger Than 30", 31 : "30+"})
df.hist(column="Salary",by="BinnedAge",bins=10)

## Barplot with groupby sort show top5
ax = sns.barplot(data=df.groupby("Team", as_index=False, sort=True).sum().sort_values(by=["Salary"], ascending=False).head(5), x="Team", y="Salary")
for item in ax.get_xticklabels():
    item.set_rotation(45)


######################### Train-test Split #########################
from sklearn.model_selection import train_test_split
Y = df["y"]
X = df.drop(["y"], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)

######################### Scaling #########################
# Use this when imputing entire dataset
numeric_attrs = ["Age", "Salary"]
from sklearn import preprocessing
def trans_num_attrs(data, numeric_attrs):
    for i in numeric_attrs: 
        scaler = preprocessing.StandardScaler()
        data[i] = scaler.fit_transform(data[[i]])
    return data
trans_num_attrs(df, numeric_attrs)
df.head()

# Use this when imputing test/train sets
train, test = train_test_split(df, shuffle=True, test_size=0.2)
def trans_num_attrs(data, data2, numeric_attrs):
    for i in numeric_attrs: 
        scaler = preprocessing.StandardScaler()
        data[i] = scaler.fit_transform(data[[i]])
        data2[i] = scaler.transform(data2[[i]])
    return data
trans_num_attrs(train, test, numeric_attrs).head()

# Or scale entire dataset if all numeric (OHE screws it up though)
from sklearn import preprocessing
s_scaler = preprocessing.StandardScaler() #MinMaxScaler() gives bad results sometimes
X_train = s_scaler.fit_transform(X_train)
X_train = s_scaler.inverse_transform(X_train)

# Scaling Test Data using Train Data Attributes
X_test_scaled = s_scaler.transform(X_test)
                 
# Min Max Scaling
numcol = ["X1", "X2"]
for col in numcol:
    df[col] = df[[col]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
  
# MinMaxScaler - Using sklearn
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns = df.columns)

######################### Saving #########################
df.to_csv("output_filename.csv", index=False)
df.to_excel("outputv2.xlsx", index=False)

######################### Other Stuff #########################
# Pivot Table
impute_grps = data.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)
print (impute_grps)

# Crosstabs
pd.crosstab(df["Team"],df["Age"],margins=True)

## Binning numeric variables into categorical
bin_labels_5 = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
df['quantile_labels'] = pd.qcut(df['ext price'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=bin_labels_5)

## Binning numeric variables into categorical, Returning threshold values as well.
results, bin_edges = pd.qcut(df['ext price'],
                            q=[0, .2, .4, .6, .8, 1],
                            labels=bin_labels_5,
                            retbins=True)
results_table = pd.DataFrame(zip(bin_edges, bin_labels_5),
                            columns=['Threshold', 'Tier'])
                 
## Importing a list of files
from glob import glob
stock_files = sorted(glob("stocks*.csv))
stock_files
pd.concat((pd.read_csv(file).assign(filename=file) for file in stock_files), ignore_index=True)
                          
## Checking for string values starting with
#https://towardsdatascience.com/learn-advanced-features-for-pythons-main-data-analysis-library-in-20-minutes-d0eedd90d086
df[col].str.endswith('ast')
df[df[col].str.contains("Bruce")]

## Mean Encoding - Melting - Similar to OHE, but using another column's value as that encoded value (Instead of 0 | 0 | 1, you get 0 | 0 | 394, in the new encoded columns)
#https://towardsdatascience.com/learn-advanced-features-for-pythons-main-data-analysis-library-in-20-minutes-d0eedd90d086
melt_experiment = pd.merge(
    invoices,
    pd.get_dummies(invoices['Type of Meal']).mul(invoices['Meal Price'].values,axis=0),
    left_index=True,
    right_index=True
)
del melt_experiment['Type of Meal']
del melt_experiment['Meal Price']
melt_experiment

## Numpy operators/ If
np.logical_or(1 ==1, 2 ==1) # and or not xor
df["Position"] = np.where(df["Position"] == "PG", "PG2",df["Position"])
df["AdjWeight"] = np.where(df["Weight"] < 200, df["Weight"]*df["Age"] , df["Weight"])
                          
## Extract first 4 digits (Data cleaning) - https://realpython.com/python-data-cleaning-numpy-pandas/
extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
                          
## Looking at duplicates
df[df.duplicated(subset = 'patient_id', keep =False)] # Find all with duplicated "patient_id" and show
repeat_patients = df.groupby(by = 'patient_id').size().sort_values(ascending =False) # Show number of repeats
df[df.duplicated(subset = 'Salary', keep =False)]["Name"].value_counts() # Prints name of those who have duplicated Salary numbers
                          
