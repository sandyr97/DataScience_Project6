# DataScience_Project6
D(St)reams of Anomalies: The real world does not slow down for bad data

Steps:
1.	Load data into data frame <br />
a.	Twitter volume <br />
2.	Clean data <br />
a.	Check for NAN values <br />
b.	Convert to datetime <br />
3.	Feature Engineering <br />
a.	Month <br />
b.	Day <br />
c.	Year <br />
d.	unixtime <br />
4.	Visualization <br />
a.	Value vs time scatter plot <br />
b.	Value distribution <br />
5.	Isolation Forest
6.	Local Outlier Factor
7.	Conclusions:

I used local outlier factor and isolation forest methods to determine the anamolies in my dataset. My dataset was used to investigate the twitter volume value for a certain date. The distribution of this dataset was definetly not a normal distrbution due to its high skewness and kurtosis values, indicating many outliers. I found that anomalies occur as the value increases. My plots for local factor outlier and isolation forest proved this observation by perdicting that the anomalies are most prevalent at values 50-100+.
