# DataScience_Project6
D(St)reams of Anomalies: The real world does not slow down for bad data

Steps:
1.	Load data into data frame
a.	Twitter volume
2.	Clean data
a.	Check for NAN values
b.	Convert to datetime
3.	Feature Engineering
a.	Month
b.	Day
c.	Year
d.	unixtime
4.	Visualization
a.	Value vs time scatter plot
b.	Value distribution
5.	Isolation Forest
6.	Local Outlier Factor
7.	Conclusions:

I used local outlier factor and isolation forest methods to determine the anamolies in my dataset. My dataset was used to investigate the twitter volume value for a certain date. The distribution of this dataset was definetly not a normal distrbution due to its high skewness and kurtosis values, indicating many outliers. I found that anomalies occur as the value increases. My plots for local factor outlier and isolation forest proved this observation by perdicting that the anomalies are most prevalent at values 50-100+.
