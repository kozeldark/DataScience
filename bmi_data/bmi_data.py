import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('bmi_data.csv')  # Read the csv file and save it to the pandas data frame.

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
g1 = sns.FacetGrid(df, col='BMI')
g1.map(plt.hist, 'Height (Inches)', bins=10)
plt.show()

# Plot weight histograms (bins=10) for each BMI value.
g2 = sns.FacetGrid(df, col='BMI')
g2.map(plt.hist, 'Weight (Pounds)', bins=10)
plt.show()

# Convert characters to numbers.
la = LabelEncoder()
la.fit(df['Sex'])
df['Sex'] = la.transform(df['Sex'])

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

# The number of NAN for each row
print("The number of NAN for each row: ")
print(df.isnull().sum(axis=1))

# The number of NAN for each column
print("The number of NAN for each column:")
print(df.isna().sum())

# Extract all rows without NAN
print("Extract all rows without NAN: ")
print(df.dropna())

# Fill NaN with mean
print("Fill NaN with mean")
print(df.fillna(df.mean()))

# Fill NaN with median
print("Fill NaN with median")
print(df.fillna(df.median()))

# Fill NaN using ffill method
print("FFill")
print(df.fillna(axis=0, method='ffill'))

# Fill NaN using bfill method
print('BFill')
print(df.fillna(axis=0, method='bfill'))


# The parentheses in the column name restrict the coding. Therefore, change the name
df.rename(columns={'Height (Inches)' : 'Height', 'Weight (Pounds)' : 'Weight'}, inplace=True)


df1 = df.dropna()  # Create a data frame with a NAN value dropped for linear regression fitting

# Assignment for Coordination
x1 = df1['Height']
y1 = df1['Weight']

# Apply data to linear regression
reg = linear_model.LinearRegression()
reg.fit(x1.values.reshape(-1, 1), y1)  # Reshape because it needs to be in two dimensions.

# Draw a line with forecasts obtained by fitting
plt.plot(x1, reg.predict(x1.values.reshape(-1, 1)))

# Coefficients and intercept Outputs obtained by fitting
print('coef', reg.coef_)
print('intercept', reg.intercept_)
# Linear function: y=a(intercept)+b(coefficient)x


# Declare a function that will replace the nan values with the predicted values when they are found
def nan2preW(k):
    if np.isnan(k.Weight):  # If nan is found in the weight column of the dataframe
        k.Weight = (reg.predict([[k.Height]]))  # Replace with forecasts obtained by linear regression
        print('NAN change to', (reg.predict([[k.Height]])))
    return k.Weight


# Linear function: y=a(intercept)+b(coefficient)x, so x = (y - a) / b
a = float(reg.intercept_)
b = float(reg.coef_)


# Predict height values through linear regression equations and weight values
def nan2preH(k):
    if np.isnan(k.Height):  # If nan is found in the height column of the dataframe
        k.Height = (float(k.Weight) - a) / b  # Replace with forecasts obtained by linear regression
        print('NAN change to', (reg.predict([[k.Weight]])))
    return k.Height


# Update nan values of data frames to predictions using the functions defined previously
df['Weight'] = df.apply(nan2preW, axis=1)
df['Height'] = df.apply(nan2preH, axis=1)

print(df)

# Output the Scatter Plot of the data frame with the nan values dropped,
# and the data frame with the nan values filled with forecasts in different colors.
x = df['Height']
y = df['Weight']
plt.title("Scatter Plot", fontsize=15)
plt.scatter(x, y, color='r')
plt.scatter(x1, y1, color='b')
plt.show()


# For male dataset
dm = pd.read_csv('bmi_data_lab3.csv')  # Read the csv file and save it to the pandas data frame.

# The parentheses in the column name restrict the coding. Therefore, change the name
dm.rename(columns={'Height (Inches)' : 'Height', 'Weight (Pounds)' : 'Weight'}, inplace=True)

dfm1 = dm.query("Sex=='Male'")  # Only import gender as male.
dm = dfm1.copy()  # Because SettingWithCopyWarning pops up, make a new copy and manage it.
dm1 = dm.dropna()  # Create a data frame with a NAN value dropped for linear regression fitting

# Assignment for Coordination
xm1 = dm1['Height']
ym1 = dm1['Weight']

# Apply data to linear regression
dm_reg = linear_model.LinearRegression()
dm_reg.fit(xm1.values.reshape(-1, 1), ym1)  # Reshape because it needs to be in two dimensions.

# Draw a line with forecasts obtained by fitting
plt.plot(xm1, dm_reg.predict(xm1.values.reshape(-1, 1)))


# Linear function: y=a(intercept)+b(coefficient)x
print('coef', dm_reg.coef_)
print('intercept', dm_reg.intercept_)


# Declare a function that will replace the nan values with the predicted values when they are found
def nan2preW_m(k):
    if np.isnan(k.Weight):  # If nan is found in the weight column of the dataframe
        k.Weight = (dm_reg.predict([[k.Height]]))  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dm_reg.predict([[k.Height]])))
    return k.Weight


# Linear function: y=a(intercept)+b(coefficient)x, so x = (y - a) / b
a = float(dm_reg.intercept_)
b = float(dm_reg.coef_)


# Predict height values through linear regression equations and weight values
def nan2preH_m(k):
    if np.isnan(k.Height):  # If nan is found in the height column of the dataframe
        k.Height = (float(k.Weight) - a) / b  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dm_reg.predict([[k.Weight]])))
    return k.Height


# Update nan values of data frames to predictions using the functions defined previously
dm['Weight'] = dm.apply(nan2preW_m, axis=1)
dm['Height'] = dm.apply(nan2preH_m, axis=1)


print(dm)

# Output the Scatter Plot of the data frame with the nan values dropped,
# and the data frame with the nan values filled with forecasts in different colors.
xm = dm['Height']
ym = dm['Weight']
plt.title("Scatter Plot(male)", fontsize=15)
plt.scatter(xm, ym, color='r')
plt.scatter(xm1, ym1, color='b')
plt.show()


# For female dataset
dff = pd.read_csv('bmi_data_lab3.csv')  # Read the csv file and save it to the pandas data frame.

# The parentheses in the column name restrict the coding. Therefore, change the name
dff.rename(columns={'Height (Inches)' : 'Height', 'Weight (Pounds)' : 'Weight'}, inplace=True)

dfm1 = dff.query("Sex=='Female'")  # Only import gender as female.
dfe = dfm1.copy()  # Because SettingWithCopyWarning pops up, make a new copy and manage it.
dfe1 = dfe.dropna()  # Create a data frame with a NAN value dropped for linear regression fitting


# Assignment for Coordination
xf1 = dfe1['Height']
yf1 = dfe1['Weight']

# Apply data to linear regression
dfe_reg = linear_model.LinearRegression()
dfe_reg.fit(xf1.values.reshape(-1, 1), yf1)  # Reshape because it needs to be in two dimensions.

# Draw a line with forecasts obtained by fitting
plt.plot(xf1, dfe_reg.predict(xf1.values.reshape(-1, 1)))

# Linear function: y=a(intercept)+b(coefficient)x
print('coef', dfe_reg.coef_)
print('intercept', dfe_reg.intercept_)


# Declare a function that will replace the nan values with the predicted values when they are found
def nan2preW_f(k):
    if np.isnan(k.Weight):  # If nan is found in the weight column of the dataframe
        k.Weight = (dfe_reg.predict([[k.Height]]))  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dfe_reg.predict([[k.Height]])))
    return k.Weight


# Linear function: y=a(intercept)+b(coefficient)x, so x = (y - a) / b
a = float(dfe_reg.intercept_)
b = float(dfe_reg.coef_)


# Predict height values through linear regression equations and weight values
def nan2preH_f(k):
    if np.isnan(k.Height):  # If nan is found in the height column of the dataframe
        k.Height = (float(k.Weight) - a) / b  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dfe_reg.predict([[k.Weight]])))
    return k.Height


# Update nan values of data frames to predictions using the functions defined previously
dfe['Weight'] = dfe.apply(nan2preW_f, axis=1)
dfe['Height'] = dfe.apply(nan2preH_f, axis=1)

print(dfe)

# Output the Scatter Plot of the data frame with the nan values dropped,
# and the data frame with the nan values filled with forecasts in different colors.
xf = dfe['Height']
yf = dfe['Weight']
plt.title("Scatter Plot(female)", fontsize=15)
plt.scatter(xf, yf, color='r')
plt.scatter(xf1, yf1, color='b')
plt.show()


# For bmi1 dataset
dfb1 = pd.read_csv('bmi_data_lab3.csv')  # Read the csv file and save it to the pandas data frame.

# The parentheses in the column name restrict the coding. Therefore, change the name
dfb1.rename(columns={'Height (Inches)' : 'Height', 'Weight (Pounds)' : 'Weight'}, inplace=True)

dfm1_1 = dfb1.query("BMI == 1")  # Only import BMI as 1.
dfbmi1 = dfm1_1.copy()  # Because SettingWithCopyWarning pops up, make a new copy and manage it.
dfbmi1_1 = dfbmi1.dropna()  # Create a data frame with a NAN value dropped for linear regression fitting

# Assignment for Coordination
xb1_1 = dfbmi1_1['Height']
yb1_1 = dfbmi1_1['Weight']

# Apply data to linear regression
dfbmi1_reg = linear_model.LinearRegression()
dfbmi1_reg.fit(xb1_1.values.reshape(-1, 1), yb1_1)

# Draw a line with forecasts obtained by fitting
plt.plot(xb1_1, dfbmi1_reg.predict(xb1_1.values.reshape(-1, 1)))

# Linear function: y=a(intercept)+b(coefficient)x
print('coef', dfbmi1_reg.coef_)
print('intercept', dfbmi1_reg.intercept_)


# Declare a function that will replace the nan values with the predicted values when they are found
def nan2preW_b1(k):
    if np.isnan(k.Weight):  # If nan is found in the weight column of the dataframe
        k.Weight = (dfbmi1_reg.predict([[k.Height]]))  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dfbmi1_reg.predict([[k.Height]])))
    return k.Weight


# Linear function: y=a(intercept)+b(coefficient)x, so x = (y - a) / b
a = float(dfbmi1_reg.intercept_)
b = float(dfbmi1_reg.coef_)


# Predict height values through linear regression equations and weight values
def nan2preH_b1(k):
    if np.isnan(k.Height):  # If nan is found in the height column of the dataframe
        k.Height = (float(k.Weight) - a) / b  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dfbmi1_reg.predict([[k.Weight]])))
    return k.Height


# Update nan values of data frames to predictions using the functions defined previously
dfbmi1['Weight'] = dfbmi1.apply(nan2preW_b1, axis=1)
dfbmi1['Height'] = dfbmi1.apply(nan2preH_b1, axis=1)

# Output the Scatter Plot of the data frame with the nan values dropped,
# and the data frame with the nan values filled with forecasts in different colors.
print(dfbmi1)
xb1 = dfbmi1['Height']
yb1 = dfbmi1['Weight']
plt.title("Scatter Plot(BMI = 1)", fontsize=15)
plt.scatter(xb1, yb1, color='r')
plt.scatter(xb1_1, yb1_1, color='b')
plt.show()


# For bmi2 dataset
dfb2 = pd.read_csv('bmi_data_lab3.csv')  # Read the csv file and save it to the pandas data frame.

# The parentheses in the column name restrict the coding. Therefore, change the name
dfb2.rename(columns={'Height (Inches)' : 'Height', 'Weight (Pounds)' : 'Weight'}, inplace=True)

dfm2_1 = dfb2.query("BMI == 2")  # Only import BMI as 2.
dfbmi2 = dfm2_1.copy()  # Because SettingWithCopyWarning pops up, make a new copy and manage it.
dfbmi2_1 = dfbmi2.dropna()  # Create a data frame with a NAN value dropped for linear regression fitting

# Assignment for Coordination
xb2_1 = dfbmi2_1['Height']
yb2_1 = dfbmi2_1['Weight']

# Apply data to linear regression
dfbmi2_reg = linear_model.LinearRegression()
dfbmi2_reg.fit(xb2_1.values.reshape(-1, 1), yb2_1)  # Reshape because it needs to be in two dimensions.

# Draw a line with forecasts obtained by fitting
plt.plot(xb2_1, dfbmi2_reg.predict(xb2_1.values.reshape(-1, 1)))

# Linear function: y=a(intercept)+b(coefficient)x
print('coef', dfbmi2_reg.coef_)
print('intercept', dfbmi2_reg.intercept_)


# Declare a function that will replace the nan values with the predicted values when they are found
def nan2preW_b2(k):
    if np.isnan(k.Weight):  # If nan is found in the weight column of the dataframe
        k.Weight = (dfbmi2_reg.predict([[k.Height]]))  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dfbmi2_reg.predict([[k.Height]])))
    return k.Weight


# Linear function: y=a(intercept)+b(coefficient)x, so x = (y - a) / b
a = float(dfbmi2_reg.intercept_)
b = float(dfbmi2_reg.coef_)


# Predict height values through linear regression equations and weight values
def nan2preH_b2(k):
    if np.isnan(k.Height):  # If nan is found in the height column of the data prime
        k.Height = (float(k.Weight) - a) / b  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dfbmi2_reg.predict([[k.Weight]])))
    return k.Height


# Update nan values of data frames to predictions using the functions defined previously
dfbmi2['Weight'] = dfbmi2.apply(nan2preW_b2, axis=1)
dfbmi2['Height'] = dfbmi2.apply(nan2preH_b2, axis=1)

print(dfbmi2)

# Output the Scatter Plot of the data frame with the nan values dropped,
# and the data frame with the nan values filled with forecasts in different colors.
xb2 = dfbmi2['Height']
yb2 = dfbmi2['Weight']
plt.title("Scatter Plot(BMI = 2)", fontsize=15)
plt.scatter(xb2, yb2, color='r')
plt.scatter(xb2_1, yb2_1, color='b')
plt.show()


# For bmi3 dataset
dfb3 = pd.read_csv('bmi_data_lab3.csv')  # Read the csv file and save it to the pandas dataframe.

# The parentheses in the column name restrict the coding. Therefore, change the name
dfb3.rename(columns={'Height (Inches)' : 'Height', 'Weight (Pounds)' : 'Weight'}, inplace=True)

dfm3_1 = dfb3.query("BMI == 3")  # Only import BMI as 3.
dfbmi3 = dfm3_1.copy()  # Because SettingWithCopyWarning pops up, make a new copy and manage it.
dfbmi3_1 = dfbmi3.dropna()  # Create a data frame with a NAN value dropped for linear regression fitting

# Assignment for Coordination
xb3_1 = dfbmi3_1['Height']
yb3_1 = dfbmi3_1['Weight']

# Apply data to linear regression
dfbmi3_reg = linear_model.LinearRegression()
dfbmi3_reg.fit(xb3_1.values.reshape(-1, 1), yb3_1)  # Reshape because it needs to be in two dimensions.

# Draw a line with forecasts obtained by fitting
plt.plot(xb3_1, dfbmi3_reg.predict(xb3_1.values.reshape(-1, 1)))

# Linear function: y=a(intercept)+b(coefficient)x

print('coef', dfbmi3_reg.coef_)
print('intercept', dfbmi3_reg.intercept_)


# Declare a function that will replace the nan values with the predicted values when they are found
def nan2preW_b3(k):
    if np.isnan(k.Weight):
        k.Weight = (dfbmi3_reg.predict([[k.Height]]))  # If nan is found in the weight column of the dataframe
        print('NAN change to', (dfbmi3_reg.predict([[k.Height]])))  # Replace with forecasts obtained by linear regression
    return k.Weight


# Linear function: y=a(intercept)+b(coefficient)x, so x = (y - a) / b
a = float(dfbmi3_reg.intercept_)
b = float(dfbmi3_reg.coef_)


# Predict height values through linear regression equations and weight values
def nan2preH_b3(k):
    if np.isnan(k.Height): # If nan is found in the height column of the data prime
        k.Height = (float(k.Weight) - a) / b  # Replace with forecasts obtained by linear regression
        print('NAN change to', (dfbmi3_reg.predict([[k.Weight]])))
    return k.Height


# Update nan values of data frames to predictions using the functions defined previously
dfbmi3['Weight'] = dfbmi3.apply(nan2preW_b3, axis=1)
dfbmi3['Height'] = dfbmi3.apply(nan2preH_b3, axis=1)

print(dfbmi3)

# Output the Scatter Plot of the data frame with the nan values dropped,
# and the data frame with the nan values filled with forecasts in different colors.
xb3 = dfbmi3['Height']
yb3 = dfbmi3['Weight']
plt.title("Scatter Plot(BMI = 3)", fontsize=15)
plt.scatter(xb3, yb3, color='r')
plt.scatter(xb3_1, yb3_1, color='b')
plt.show()



# compare the replacement values computed using different regression equations

dfcp = pd.read_csv('bmi_data_lab3.csv')  # Read the csv file and save it to the pandas dataframe.

# The parentheses in the column name restrict the coding. Therefore, change the name
dfcp.rename(columns={'Height (Inches)' : 'Height', 'Weight (Pounds)' : 'Weight'}, inplace=True)


dfcp1 = dfcp.dropna()  # Create a data frame with a NAN value dropped for linear regression fitting

# Assignment for Coordination
xc1 = dfcp1['Height']
yc1 = dfcp1['Weight']

# Apply data to linear regression
reg = linear_model.LinearRegression()
reg.fit(xc1.values.reshape(-1, 1), yc1)  # Reshape because it needs to be in two dimensions.

# Draw a line with forecasts obtained by fitting
plt.plot(xc1, reg.predict(xc1.values.reshape(-1, 1)))


# Getting weight is all the same, based on the d that initially used all the datasets.
# The height here is based on the dm that used only male data.
print('-d for dm-')
print('-weight-')
dfcp['Weight'] = dfcp.apply(nan2preW, axis=1)
print('-height-')
dfcp['Height'] = dfcp.apply(nan2preH_m, axis=1)

dfcp = pd.read_csv('bmi_data_lab3.csv')  # Initialization
# The parentheses in the column name restrict the coding. Therefore, change the name
dfcp.rename(columns={'Height (Inches)' : 'Height', 'Weight (Pounds)' : 'Weight'}, inplace=True)


# The height here is based on the dm that used only female data.
print('d for df')
print('-weight-')
dfcp['Weight'] = dfcp.apply(nan2preW, axis=1)
print('-height-')
dfcp['Height'] = dfcp.apply(nan2preH_f, axis=1)


