import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import time
import timeit

# Read the csv file and divide it by the x,y axis
mnist = pd.read_csv('fashion-mnist.csv')# Read the csv file and save it to the pandas data frame.
train_x = mnist[list(mnist.columns)[1:]].values
train_y = mnist['label'].values
print('---fashion-mnist dataset---')
print(mnist)

# After scaling with the standard scaler, print out only the first 30 lines
stScaler = preprocessing.StandardScaler()
train_stscale = stScaler.fit_transform(mnist)
train_stscale = pd.DataFrame(train_stscale)
print('---After Standard Scaler(first 30 lines)---')
print(train_stscale.iloc[0:30])

# After scaling with the min-max scaler, print out only the first 30 lines
mmScaler = preprocessing.MinMaxScaler()
train_mmscale = mmScaler.fit_transform(mnist)
train_mmscale = pd.DataFrame(train_mmscale)
print('---After Min-Max Scaler(first 30 lines)---')
print(train_mmscale.iloc[0:30])

# After scaling with the max-abs scaler, print out only the first 30 lines
maScaler = preprocessing.MaxAbsScaler()
train_mascale = maScaler.fit_transform(mnist)
train_mascale = pd.DataFrame(train_mascale)
print('---After Max-Abs Scaler(first 30 lines)---')
print(train_mascale.iloc[0:30])

# After scaling with the robust scaler, print out only the first 30 lines
rbScaler = preprocessing.RobustScaler()
train_rbscale = rbScaler.fit_transform(mnist)
train_rbscale = pd.DataFrame(train_rbscale)
print('---After Robust Scaler(first 30 lines)---')
print(train_rbscale.iloc[0:30])

# Divide dataset into original and test uses
train_img, test_img, train_lbl, test_lbl = train_test_split(train_x, train_y, test_size=1/7.0, random_state=0)

#perform PCA transformation on both datasets such that 80% of variance are retained.
pca = PCA(.80)
pca.fit(train_img)
train_img80 = pca.transform(train_img)
test_img80 = pca.transform(test_img)
noc80 = pca.n_components_  # number of components

#perform PCA transformation on both datasets such that 85% of variance are retained.
pca1 = PCA(.85)
pca1.fit(train_img)
train_img85 = pca1.transform(train_img)
test_img85 = pca1.transform(test_img)
noc85 = pca.n_components_  # number of components

#perform PCA transformation on both datasets such that 90% of variance are retained.
pca = PCA(.90)
pca.fit(train_img)
train_img90 = pca.transform(train_img)
test_img90 = pca.transform(test_img)
noc90 = pca.n_components_  # number of components

#perform PCA transformation on both datasets such that 95% of variance are retained.
pca = PCA(.95)
pca.fit(train_img)
train_img95 = pca.transform(train_img)
test_img95 = pca.transform(test_img)
noc95 = pca.n_components_  # number of components

#perform PCA transformation on both datasets such that 99% of variance are retained.
pca = PCA(.99)
pca.fit(train_img)
train_img99 = pca.transform(train_img)
test_img99 = pca.transform(test_img)
noc99 = pca.n_components_  # number of components

#perform PCA transformation on both datasets such that 100%(28 x 28 = 784) of variance are retained.
pca = PCA(784)
pca.fit(train_img)
train_img100 = pca.transform(train_img)
test_img100 = pca.transform(test_img)
noc100 = pca.n_components_  # number of components

# LogisticRegression for 80% of variance are retained.
start_time = time.time()  # Set Start Time for Time Measurement
#Using solver as liblinear and increasing max_iter further, warning did not appear, but it took more than 30 minutes for one dataset, so default value.
model = LogisticRegression()
model.fit(train_img80, train_lbl)
prediction = model.predict(test_img80[0].reshape(1, -1))
end_time = time.time()  # Set End Time for Time Measurement
score80 = model.score(train_img80, train_lbl)  # accuracy
time80 = end_time - start_time  # runnung time

# LogisticRegression for 85% of variance are retained.
start_time = time.time()
model.fit(train_img85, train_lbl)
prediction = model.predict(test_img85[0].reshape(1, -1))
end_time = time.time()
score85 = model.score(train_img85, train_lbl)
time85 = end_time - start_time

# LogisticRegression for 90% of variance are retained.
start_time = time.time()
model.fit(train_img90, train_lbl)
prediction = model.predict(train_img90[0].reshape(1, -1))
end_time = time.time()
score90 = model.score(train_img90, train_lbl)
time90 = end_time - start_time

# LogisticRegression for 95% of variance are retained.
start_time = time.time()
model.fit(train_img95, train_lbl)
prediction = model.predict(train_img95[0].reshape(1, -1))
end_time = time.time()
score95 = model.score(train_img95, train_lbl)
time95 = end_time - start_time

# LogisticRegression for 99% of variance are retained.
start_time = time.time()
model.fit(train_img99, train_lbl)
prediction = model.predict(train_img99[0].reshape(1, -1))
end_time = time.time()
score99 = model.score(train_img99, train_lbl)
time99 = end_time - start_time

# LogisticRegression for 100% of variance are retained.
start_time = time.time()
model.fit(train_img100, train_lbl)
prediction = model.predict(train_img100[0].reshape(1, -1))
end_time = time.time()
score100 = model.score(train_img100, train_lbl)
time100 = end_time - start_time

# Print a table Performance Result from original (unscaled) dataset
col = ["Variance Rated", "Number of Components", "Time(seconds)", "Accuracy"]
ind = [0, 1, 2, 3, 4, 5]
con = [[1.00, noc100, time100, score100], [0.99, noc99, time99, score99], [0.95, noc95, time95, score95],
       [0.90, noc90, time90, score90], [0.85, noc85, time85, score85], [0.80, noc80, time80, score80]]

result = pd.DataFrame(con, columns=col, index=ind)
print('---Performance Result(original (unscaled) dataset)---')
print(result)


# Standard Scaling dataset
train_xst = train_stscale[list(train_stscale.columns)[1:]].values
train_imgst, test_imgst, train_lbl, test_lbl = train_test_split(train_xst, train_y, test_size=1/7.0, random_state=0)

pca = PCA(.80)
pca.fit(train_imgst)
train_imgst80 = pca.transform(train_imgst)
test_imgst80 = pca.transform(test_imgst)
noc80st = pca.n_components_

pca1 = PCA(.85)
pca1.fit(train_imgst)
train_imgst85 = pca1.transform(train_imgst)
test_imgst85 = pca1.transform(test_imgst)
noc85st = pca.n_components_

pca = PCA(.90)
pca.fit(train_imgst)
train_imgst90 = pca.transform(train_imgst)
test_imgst90 = pca.transform(test_imgst)
noc90st = pca.n_components_

pca = PCA(.95)
pca.fit(train_imgst)
train_imgst95 = pca.transform(train_imgst)
test_imgst95 = pca.transform(test_imgst)
noc95st = pca.n_components_

pca = PCA(.99)
pca.fit(train_imgst)
train_imgst99 = pca.transform(train_imgst)
test_imgst99 = pca.transform(test_imgst)
noc99st = pca.n_components_

pca = PCA(784)
pca.fit(train_imgst)
train_imgst100 = pca.transform(train_imgst)
test_imgst100 = pca.transform(test_imgst)
noc100st = pca.n_components_


start_time = time.time()
model = LogisticRegression()
model.fit(train_imgst80, train_lbl)
prediction = model.predict(test_imgst80[0].reshape(1, -1))
end_time = time.time()
score80st = model.score(train_imgst80, train_lbl)
time80st = end_time - start_time


start_time = time.time()
model.fit(train_imgst85, train_lbl)
prediction = model.predict(test_imgst85[0].reshape(1, -1))
end_time = time.time()
score85st = model.score(train_imgst85, train_lbl)
time85st = end_time - start_time


start_time = time.time()
model.fit(train_imgst90, train_lbl)
prediction = model.predict(train_imgst90[0].reshape(1, -1))
end_time = time.time()
score90st = model.score(train_imgst90, train_lbl)
time90st = end_time - start_time


start_time = time.time()
model.fit(train_imgst95, train_lbl)
prediction = model.predict(train_imgst95[0].reshape(1, -1))
end_time = time.time()
score95st = model.score(train_imgst95, train_lbl)
time95st = end_time - start_time


start_time = time.time()
model.fit(train_imgst99, train_lbl)
prediction = model.predict(train_imgst99[0].reshape(1, -1))
end_time = time.time()
score99st = model.score(train_imgst99, train_lbl)
time99st = end_time - start_time


start_time = time.time()
model.fit(train_imgst100, train_lbl)
prediction = model.predict(train_imgst100[0].reshape(1, -1))
end_time = time.time()
score100st = model.score(train_imgst100, train_lbl)
time100st = end_time - start_time

# Print a table Performance Result from Standard Scaling dataset

colst = ["Variance Rated", "Number of Components", "Time(seconds)", "Accuracy"]
indst = [0, 1, 2, 3, 4, 5]
const = [[1.00, noc100st, time100st, score100st], [0.99, noc99st, time99st, score99st], [0.95, noc95st, time95st, score95st],
       [0.90, noc90st, time90st, score90st], [0.85, noc85st, time85st, score85st], [0.80, noc80st, time80st, score80st]]

resultst = pd.DataFrame(const, columns=colst, index=indst)
print('---Performance Result(Standard Scaler dataset)---')
print(resultst)

# Min-max Scaling dataset
train_xmm = train_mmscale[list(train_mmscale.columns)[1:]].values
train_imgmm, test_imgmm, train_lbl, test_lbl = train_test_split(train_xmm, train_y, test_size=1/7.0, random_state=0)

pca = PCA(.80)
pca.fit(train_imgmm)
train_imgmm80 = pca.transform(train_imgmm)
test_imgmm80 = pca.transform(test_imgmm)
noc80mm = pca.n_components_

pca1 = PCA(.85)
pca1.fit(train_imgmm)
train_imgmm85 = pca1.transform(train_imgmm)
test_imgmm85 = pca1.transform(test_imgmm)
noc85mm = pca.n_components_

pca = PCA(.90)
pca.fit(train_imgmm)
train_imgmm90 = pca.transform(train_imgmm)
test_imgmm90 = pca.transform(test_imgmm)
noc90mm = pca.n_components_

pca = PCA(.95)
pca.fit(train_imgmm)
train_imgmm95 = pca.transform(train_imgmm)
test_imgmm95 = pca.transform(test_imgmm)
noc95mm = pca.n_components_

pca = PCA(.99)
pca.fit(train_imgmm)
train_imgmm99 = pca.transform(train_imgmm)
test_imgmm99 = pca.transform(test_imgmm)
noc99mm = pca.n_components_

pca = PCA(784)
pca.fit(train_imgmm)
train_imgmm100 = pca.transform(train_imgmm)
test_imgmm100 = pca.transform(test_imgmm)
noc100mm = pca.n_components_


start_time = time.time()
model = LogisticRegression()
model.fit(train_imgmm80, train_lbl)
prediction = model.predict(test_imgmm80[0].reshape(1, -1))
end_time = time.time()
score80mm = model.score(train_imgmm80, train_lbl)
time80mm = end_time - start_time


start_time = time.time()
model.fit(train_imgmm85, train_lbl)
prediction = model.predict(test_imgmm85[0].reshape(1, -1))
end_time = time.time()
score85mm = model.score(train_imgmm85, train_lbl)
time85mm = end_time - start_time


start_time = time.time()
model.fit(train_imgmm90, train_lbl)
prediction = model.predict(train_imgmm90[0].reshape(1, -1))
end_time = time.time()
score90mm = model.score(train_imgmm90, train_lbl)
time90mm = end_time - start_time


start_time = time.time()
model.fit(train_imgmm95, train_lbl)
prediction = model.predict(train_imgmm95[0].reshape(1, -1))
end_time = time.time()
score95mm = model.score(train_imgmm95, train_lbl)
time95mm = end_time - start_time


start_time = time.time()
model.fit(train_imgmm99, train_lbl)
prediction = model.predict(train_imgmm99[0].reshape(1, -1))
end_time = time.time()
score99mm = model.score(train_imgmm99, train_lbl)
time99mm = end_time - start_time


start_time = time.time()
model.fit(train_imgmm100, train_lbl)
prediction = model.predict(train_imgmm100[0].reshape(1, -1))
end_time = time.time()
score100mm = model.score(train_imgmm100, train_lbl)
time100mm = end_time - start_time

# Print a table Performance Result from Min-Max Scaling dataset
colmm = ["Variance Rated", "Number of Components", "Time(seconds)", "Accuracy"]
indmm = [0, 1, 2, 3, 4, 5]
conmm = [[1.00, noc100mm, time100mm, score100mm], [0.99, noc99mm, time99mm, score99mm], [0.95, noc95mm, time95mm, score95mm],
       [0.90, noc90mm, time90mm, score90mm], [0.85, noc85mm, time85mm, score85mm], [0.80, noc80mm, time80mm, score80mm]]

resultmm = pd.DataFrame(conmm, columns=colmm, index=indmm)
print('---Performance Result(Min-Max Scaler dataset)---')
print(resultmm)

# Max-Abs Scaling dataset
train_xma = train_mascale[list(train_mascale.columns)[1:]].values
train_imgma, test_imgma, train_lbl, test_lbl = train_test_split(train_xma, train_y, test_size=1/7.0, random_state=0)

pca = PCA(.80)
pca.fit(train_imgma)
train_imgma80 = pca.transform(train_imgma)
test_imgma80 = pca.transform(test_imgma)
noc80ma = pca.n_components_

pca1 = PCA(.85)
pca1.fit(train_imgma)
train_imgma85 = pca1.transform(train_imgma)
test_imgma85 = pca1.transform(test_imgma)
noc85ma = pca.n_components_

pca = PCA(.90)
pca.fit(train_imgma)
train_imgma90 = pca.transform(train_imgma)
test_imgma90 = pca.transform(test_imgma)
noc90ma = pca.n_components_

pca = PCA(.95)
pca.fit(train_imgma)
train_imgma95 = pca.transform(train_imgma)
test_imgma95 = pca.transform(test_imgma)
noc95ma = pca.n_components_

pca = PCA(.99)
pca.fit(train_imgma)
train_imgma99 = pca.transform(train_imgma)
test_imgma99 = pca.transform(test_imgma)
noc99ma = pca.n_components_

pca = PCA(784)
pca.fit(train_imgma)
train_imgma100 = pca.transform(train_imgma)
test_imgma100 = pca.transform(test_imgma)
noc100ma = pca.n_components_


start_time = time.time()
model = LogisticRegression()
model.fit(train_imgma80, train_lbl)
prediction = model.predict(test_imgma80[0].reshape(1, -1))
end_time = time.time()
score80ma = model.score(train_imgma80, train_lbl)
time80ma = end_time - start_time


start_time = time.time()
model.fit(train_imgma85, train_lbl)
prediction = model.predict(test_imgma85[0].reshape(1, -1))
end_time = time.time()
score85ma = model.score(train_imgma85, train_lbl)
time85ma = end_time - start_time


start_time = time.time()
model.fit(train_imgma90, train_lbl)
prediction = model.predict(train_imgma90[0].reshape(1, -1))
end_time = time.time()
score90ma = model.score(train_imgma90, train_lbl)
time90ma = end_time - start_time


start_time = time.time()
model.fit(train_imgma95, train_lbl)
prediction = model.predict(train_imgma95[0].reshape(1, -1))
end_time = time.time()
score95ma = model.score(train_imgma95, train_lbl)
time95ma = end_time - start_time


start_time = time.time()
model.fit(train_imgma99, train_lbl)
prediction = model.predict(train_imgma99[0].reshape(1, -1))
end_time = time.time()
score99ma = model.score(train_imgma99, train_lbl)
time99ma = end_time - start_time


start_time = time.time()
model.fit(train_imgma100, train_lbl)
prediction = model.predict(train_imgma100[0].reshape(1, -1))
end_time = time.time()
score100ma = model.score(train_imgma100, train_lbl)
time100ma = end_time - start_time

# Print a table Performance Result from Max-Abs Scaling dataset
colma = ["Variance Rated", "Number of Components", "Time(seconds)", "Accuracy"]
indma = [0, 1, 2, 3, 4, 5]
conma = [[1.00, noc100ma, time100ma, score100ma], [0.99, noc99ma, time99ma, score99ma], [0.95, noc95ma, time95ma, score95ma],
       [0.90, noc90ma, time90ma, score90ma], [0.85, noc85ma, time85ma, score85ma], [0.80, noc80ma, time80ma, score80ma]]

resultma = pd.DataFrame(conma, columns=colma, index=indma)
print('---Performance Result(Max-Abs Scaler dataset)---')
print(resultma)

# Robust Scaling dataset
train_xrb = train_rbscale[list(train_rbscale.columns)[1:]].values
train_imgrb, test_imgrb, train_lbl, test_lbl = train_test_split(train_xrb, train_y, test_size=1/7.0, random_state=0)

pca = PCA(.80)
pca.fit(train_imgrb)
train_imgrb80 = pca.transform(train_imgrb)
test_imgrb80 = pca.transform(test_imgrb)
noc80rb = pca.n_components_

pca1 = PCA(.85)
pca1.fit(train_imgrb)
train_imgrb85 = pca1.transform(train_imgrb)
test_imgrb85 = pca1.transform(test_imgrb)
noc85rb = pca.n_components_

pca = PCA(.90)
pca.fit(train_imgrb)
train_imgrb90 = pca.transform(train_imgrb)
test_imgrb90 = pca.transform(test_imgrb)
noc90rb = pca.n_components_

pca = PCA(.95)
pca.fit(train_imgrb)
train_imgrb95 = pca.transform(train_imgrb)
test_imgrb95 = pca.transform(test_imgrb)
noc95rb = pca.n_components_

pca = PCA(.99)
pca.fit(train_imgrb)
train_imgrb99 = pca.transform(train_imgrb)
test_imgrb99 = pca.transform(test_imgrb)
noc99rb = pca.n_components_

pca = PCA(784)
pca.fit(train_imgrb)
train_imgrb100 = pca.transform(train_imgrb)
test_imgrb100 = pca.transform(test_imgrb)
noc100rb = pca.n_components_


start_time = time.time()
model = LogisticRegression()
model.fit(train_imgrb80, train_lbl)
prediction = model.predict(test_imgrb80[0].reshape(1, -1))
end_time = time.time()
score80rb = model.score(train_imgrb80, train_lbl)
time80rb = end_time - start_time


start_time = time.time()
model.fit(train_imgrb85, train_lbl)
prediction = model.predict(test_imgrb85[0].reshape(1, -1))
end_time = time.time()
score85rb = model.score(train_imgrb85, train_lbl)
time85rb = end_time - start_time


start_time = time.time()
model.fit(train_imgrb90, train_lbl)
prediction = model.predict(train_imgrb90[0].reshape(1, -1))
end_time = time.time()
score90rb = model.score(train_imgrb90, train_lbl)
time90rb = end_time - start_time


start_time = time.time()
model.fit(train_imgrb95, train_lbl)
prediction = model.predict(train_imgrb95[0].reshape(1, -1))
end_time = time.time()
score95rb = model.score(train_imgrb95, train_lbl)
time95rb = end_time - start_time


start_time = time.time()
model.fit(train_imgrb99, train_lbl)
prediction = model.predict(train_imgrb99[0].reshape(1, -1))
end_time = time.time()
score99rb = model.score(train_imgrb99, train_lbl)
time99rb = end_time - start_time


start_time = time.time()
model.fit(train_imgrb100, train_lbl)
prediction = model.predict(train_imgrb100[0].reshape(1, -1))
end_time = time.time()
score100rb = model.score(train_imgrb100, train_lbl)
time100rb = end_time - start_time

# Print a table Performance Result from Robust Scaling dataset
colrb = ["Variance Rated", "Number of Components", "Time(seconds)", "Accuracy"]
indrb = [0, 1, 2, 3, 4, 5]
conrb = [[1.00, noc100rb, time100rb, score100rb], [0.99, noc99rb, time99rb, score99rb], [0.95, noc95rb, time95rb, score95rb],
       [0.90, noc90rb, time90rb, score90rb], [0.85, noc85rb, time85rb, score85rb], [0.80, noc80rb, time80rb, score80rb]]

resultrb = pd.DataFrame(conrb, columns=colrb, index=indrb)
print('---Performance Result(Robust Scaler dataset)---')
print(resultrb)
