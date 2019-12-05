pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("../data/Twitter_volume_FB.csv")
df.head(10)

df.isnull().sum()

df['timestamp'] = pd.to_datetime(df['timestamp'])
print(df.dtypes)
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year
print(df.head(10))
dt=df['timestamp']
dt = pd.DatetimeIndex ( dt ).astype ( np.int64 )/1000000
df['unixTime']=dt
print(df.head(10))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticklabels(df['timestamp'], rotation=70)
ax.plot_date(x=df.timestamp, y=df.value, ls='-', marker='.')
import sklearn as sc
from sklearn.model_selection import train_test_split
labels = df['value']
features = df[['timestamp']]
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size=0.20,
                                                    random_state=42)
from sklearn.ensemble import IsolationForest

model = IsolationForest()
model.fit(X_train, y_train)
#Predicting the label of the new data set
y_pred = model.predict(X_test)
print (y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
import seaborn as sns
sns.distplot(df['value'])
plt.title("Distribution of Values")
sns.despine()
print("Skewness: %f" % df['value'].skew())
print("Kurtosis: %f" % df['value'].kurt())

isolation_forest = IsolationForest(n_estimators=100)
isolation_forest.fit(df['value'].values.reshape(-1, 1))
xx = np.linspace(df['value'].min(), df['value'].max(), len(df)).reshape(-1,1)
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)
plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                 where=outlier==-1, color='r',
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('Value')
plt.show();

X=np.array(df['value']).reshape(-1, 1)
rs=np.random.RandomState(0)
clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1)
clf.fit(X)
if_scores = clf.decision_function(X)
if_anomalies=clf.predict(X)
if_anomalies=pd.Series(if_anomalies).replace([-1,1],[1,0])
if_anomalies=df[if_anomalies==1]

plt.figure(figsize=(12,8))
plt.hist(if_scores)
plt.title('Histogram of Avg Anomaly Scores: Lower => More Anomalous')

def LOF_plot(k):
 import seaborn as sns
 from sklearn.neighbors import LocalOutlierFactor
 var1,var2=1,2
 clf = LocalOutlierFactor(n_neighbors=k, contamination=.1)
 y_pred = clf.fit_predict(X)
 LOF_Scores = clf.negative_outlier_factor_

 plt.title("Local Outlier Factor (LOF), K={}".format(k))
 plt.scatter(df.iloc[:, 5], df.iloc[:, 1], color='k', s=3., label='Data points')
 radius = (LOF_Scores.max() - LOF_Scores) / (LOF_Scores.max() - LOF_Scores.min())
 plt.scatter(df.iloc[:, 5],df.iloc[:, 1], s=1000 * radius, edgecolors='r',
 facecolors='none', label='Outlier Score')
 plt.axis('tight')
 plt.ylabel("{}".format(df.columns[1]))
 plt.xlabel("{}".format(df.columns[5]))
 legend = plt.legend(loc='upper right')
 legend.legendHandles[0]._sizes = [10]
 legend.legendHandles[1]._sizes = [20]
 plt.ylim(0, 400)
 plt.show()
LOF_plot(5)
LOF_plot(30)
LOF_plot(70)
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=30, contamination=.1)
y_pred = clf.fit_predict(X)
LOF_Scores = clf.negative_outlier_factor_
LOF_pred=pd.Series(y_pred).replace([-1,1],[1,0])
LOF_anomalies=df[LOF_pred==1]
cmap=np.array(['white','red'])
plt.scatter(df.iloc[:,5],df.iloc[:,1],c='white',s=20,edgecolor='k', label='Data points')
plt.scatter(LOF_anomalies.iloc[:,5],LOF_anomalies.iloc[:,1],c='red', label='Anomalies')
 #,marker=’x’,s=100)
plt.title('Local Outlier Factor-Anomalies')
plt.xlabel('Time')
plt.ylabel('value')
plt.ylim(0, 400)
plt.xlim(1.425e+12, 1.42516e+12)
legend = plt.legend(loc='upper right')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]

cmap=np.array(['white','red'])
plt.scatter(df.iloc[:,5], df.iloc[:,1],c='white',s=20,edgecolor='k', label='Data points')
plt.scatter(if_anomalies.iloc[:,5],if_anomalies.iloc[:,1],c='red', label='Anomalies')
legend = plt.legend(loc='upper right')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Isolation Forests - Anomalies')
plt.ylim(0, 400)
plt.xlim(1.425e+12, 1.42516e+12)

