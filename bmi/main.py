import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('bmi_data_phw3.xlsx')  # Read the xlsx file and save it to the pandas data frame.

# Print dataset statistical data.
print("Statistical data: ")
print(df.describe())

# Print feature names.
print("Feature names: ")
print(df.columns.values)

# Print data types.
print("Data types: ")
print(df.dtypes)

# Plot height histograms (bins=10) for each BMI value.
grid = sns.FacetGrid(df, col='BMI')
grid.map(plt.hist, 'Height (Inches)', bins=10)
plt.show()

# Plot weight histograms (bins=10) for each BMI value.
grid1 = sns.FacetGrid(df, col='BMI')
grid1.map(plt.hist, 'Weight (Pounds)', bins=10)
plt.show()

# Convert characters to numbers.(sex(male, female) to 0, 1)
LabelEncoder = LabelEncoder()
LabelEncoder.fit(df['Sex'])
df['Sex'] = LabelEncoder.transform(df['Sex'])

# Normalize data frames to Standard Scaler
stScaler = preprocessing.StandardScaler()
stscaled_df = stScaler.fit_transform(df)
stscaled_df = pd.DataFrame(stscaled_df, columns=['Sex', 'Age', 'Height (Inches)', 'Weight (Pounds)', 'BMI'])

# Normalize data frames to Min-Max Scaler
mmScaler = preprocessing.MinMaxScaler()
mmscaled_df = mmScaler.fit_transform(df)
mmscaled_df = pd.DataFrame(mmscaled_df, columns=['Sex', 'Age', 'Height (Inches)', 'Weight (Pounds)', 'BMI'])

# Normalize data frames to Robust Scaler
rbtScaler = preprocessing.RobustScaler()
rbscaled_df = rbtScaler.fit_transform(df)
rbscaled_df = pd.DataFrame(rbscaled_df, columns=['Sex', 'Age', 'Height (Inches)', 'Weight (Pounds)', 'BMI'])

# 4 plot outputs previously scaled by height
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(10, 5))  # Create subplot for outputting 4 plots
ax1.set_title('Before Scaling')
sns.kdeplot(df['Height (Inches)'], ax=ax1)  # Plot scaling results for height
ax2.set_title('After Standard Scaling')
sns.kdeplot(stscaled_df['Height (Inches)'], ax=ax2)  # Standard scaling results for height
ax3.set_title('After MinMax Scaling')
sns.kdeplot(mmscaled_df['Height (Inches)'], ax=ax3)  # Min-Max scaling results for height
ax4.set_title('After Robust Scaling')
sns.kdeplot(rbscaled_df['Height (Inches)'], ax=ax4)  # Robust scaling results for height
plt.show()

# 4 plot outputs previously scaled by weight
fig, (ax5, ax6, ax7, ax8) = plt.subplots(ncols=4, figsize=(10, 5))  # Create subplot for outputting 4 plots
ax5.set_title('Before Scaling')
sns.kdeplot(df['Weight (Pounds)'], ax=ax5)  # Plot scaling results for weight
ax6.set_title('After Standard Scaling')
sns.kdeplot(stscaled_df['Weight (Pounds)'], ax=ax6)  # Standard scaling results for weight
ax7.set_title('After MinMax Scaling')
sns.kdeplot(mmscaled_df['Weight (Pounds)'], ax=ax7)  # Min-Max scaling results for weight
ax8.set_title('After Robust Scaling')
sns.kdeplot(rbscaled_df['Weight (Pounds)'], ax=ax8)  # Robust scaling results for weight
plt.show()

# The parentheses in the column name restrict the coding. Therefore, change the name
df.rename(columns={'Height (Inches)': 'Height', 'Weight (Pounds)': 'Weight'}, inplace=True)

# Assignment for Coordination
x = df['Height']
y = df['Weight']

# Apply data to linear regression
reg = linear_model.LinearRegression()
reg.fit(x.values.reshape(-1, 1), y)  # Reshape because it needs to be in two dimensions.

# Create new columns with predictions obtained by linear regression
df['Weight(predict)'] = df.apply(lambda x: float(reg.predict([[x.Height]])), axis=1)

# Normalize the e values
normalize = []
e = df['Weight'] - df['Weight(predict)']  # Calculate linear regression equations from predictions
for value in e:
    ze = (value - np.mean(e)) / np.std(e)
    normalize.append(ze)

# plot a histogram showing the distribution of ze (~10 bins)
plt.hist(normalize, bins=10)
plt.title('Weight Reference Histogram')
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()

# Decide for records with ze, set BMI
bmi = []
for i in normalize:
    if i < 0:
        bmi.append(0)
        continue
    else:
        bmi.append(4)
df['BMI(predict)'] = bmi


print(df)  # Final Result Data Frame Output

# For male dataset
dfm = pd.read_excel('bmi_data_phw3.xlsx')  # Read the xlsx file and save it to the pandas data frame.

# The parentheses in the column name restrict the coding. Therefore, change the name
dfm.rename(columns={'Height (Inches)': 'Height', 'Weight (Pounds)': 'Weight'}, inplace=True)

dfm1 = dfm.query("Sex=='Male'")  # Only import gender as male.
dm = dfm1.copy()  # Because SettingWithCopyWarning pops up, make a new copy and manage it.

# Assignment for Coordination
dm_x = dm['Height']
dm_y = dm['Weight']

# Apply data to linear regression
dm_reg = linear_model.LinearRegression()
dm_reg.fit(dm_x.values.reshape(-1, 1), dm_y)  # Reshape because it needs to be in two dimensions.

# Create new columns with predictions obtained by linear regression
dm['Weight(predict)'] = dfm1.apply(lambda x: float(dm_reg.predict([[x.Height]])), axis=1)


# Normalize the e values
normalize_m = []
dm['e'] = dm['Weight'] - dm['Weight(predict)']
for value in dm['e']:
    ze_m = (value - np.mean(dm['e'])) / np.std(dm['e'])
    normalize_m.append(ze_m)

# plot a histogram showing the distribution of ze (~10 bins)
plt.hist(normalize_m, bins=10)
plt.title("Weight Reference Histogram(only male)")
plt.xlabel('ze')
plt.ylabel('frequency')
plt.show()

# Decide for records with ze, set BMI
bmi_m = []
for i in normalize_m:
    if i < 0:
        bmi_m.append(0)
        continue
    else:
        bmi_m.append(4)
dm['BMI(predict)'] = bmi_m

print(dm)  # Final Result Data Frame Output

# For female dataset
dff = pd.read_excel('bmi_data_phw3.xlsx')  # Read the xlsx file and save it to the pandas data frame.

# The parentheses in the column name restrict the coding. Therefore, change the name
dff.rename(columns={'Height (Inches)': 'Height', 'Weight (Pounds)': 'Weight'}, inplace=True)

dff1 = dff.query("Sex=='Female'")  # Only import gender as female.
dfe = dff1.copy()  # Because SettingWithCopyWarning pops up, make a new copy and manage it.

# Assignment for Coordination
dfe_x = dfe['Height']
dfe_y = dfe['Weight']

# Apply data to linear regression
dfe_reg = linear_model.LinearRegression()
dfe_reg.fit(dfe_x.values.reshape(-1, 1), dfe_y)  # Reshape because it needs to be in two dimensions.


dfe['Weight(predict)'] = dff1.apply(lambda x: float(dfe_reg.predict([[x.Height]])), axis=1)  # predict the female's weight using the equation

dfe_e = dfe['Weight'] - dfe['Weight(predict)']  # Compute female's e by subtracting original weight with predicted weight

normalize_f = []
# Normalized Data
for value in dfe_e:
    ze_f = (value - np.mean(dfe_e)) / np.std(dfe_e)
    normalize_f.append(ze_f)

# plot a histogram showing the distribution of ze (~10 bins)
plt.hist(normalize_f, bins=10)
plt.title("Weight Reference Histogram(only female)")
plt.xlabel('ze')
plt.ylabel('frequency')
plt.show()

# Decide for records with ze, set BMI
bmi_f = []
for i in normalize_f:
    if i < 0:
        bmi_f.append(0)
        continue
    else:
        bmi_f.append(4)
dfe['BMI(predict)'] = bmi_f

print(dfe)  # Final Result Data Frame Output

