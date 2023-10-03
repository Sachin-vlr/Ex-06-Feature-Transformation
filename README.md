# Ex-06-Feature-Transformation

## AIM:

To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:

1. Read the given data

2. Clean the Data Set using Data Cleaning Process

3. Apply Feature Transformation techniques to all the features of the data set

4. Save the data to the file

## PROGRAM:

```PYTHON
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```

## OUTPUT:

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/6ca72e01-5447-4ced-b19b-7ee57992ce27)

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/c55f97ed-49cf-4f24-8e3b-cf6796d09b01)

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/34e5c1cf-92d3-42f1-9f35-c0114652d4a1)

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/e079c212-8cd3-46fc-b2ff-9e737a898ae3)

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/b4d15240-ba62-48b5-9146-c992f6a1f134)

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/2f2f42fd-a1a7-4dbc-8d41-fdb5f1fcb588)

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/87b9e316-1e47-4ae6-8a8c-d43b35c6ccad)

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/fc53866b-6daf-405a-bed5-15e70081c7da)

![image](https://github.com/Sachin-vlr/Ex-06-Feature-Transformation/assets/113497666/3d5349bf-52b4-4db5-aa2a-4d39d9e1131f)


## RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully.

  

