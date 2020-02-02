#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.gridspec as gridspec
from numpy import random
import scipy.stats as scipy

from IPython.display import display

from sklearn.metrics import accuracy_score


# In[2]:


pip install xlrd


# In[3]:


nfl = pd.ExcelFile('/Users/shannonmayjusman/Downloads/2019 Regular Season Data.xlsx')
df_nfl = nfl.parse('Sheet1')


# In[4]:


df_nfl


# In[5]:


feature_table = df_nfl[['Team1','Team2', 'Team1_Score', 'Team2_Score', 'Team1_Passing', 'Team1_Rushing', 'Team1_Turnovers', 'Team2_Passing', 'Team2_Rushing', 'Team2_Turnovers']]


# In[6]:


feature_table


# In[7]:


def result(row):
    if row.Team1_Score > row.Team2_Score:
        return 1
    elif row.Team1_Score < row.Team2_Score:
        return -1
    else:
        return 0


# In[8]:


feature_table["Winner"] = feature_table.apply(lambda row: result(row),axis=1)


# In[9]:


feature_table


# In[10]:


X = feature_table[['Team1_Score', 'Team2_Score', 'Team1_Passing', 'Team1_Rushing', 'Team1_Turnovers', 'Team2_Passing', 'Team2_Rushing', 'Team2_Turnovers']]
y = feature_table['Winner']


# In[11]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[13]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[14]:


y_pred = regressor.predict(X_test)


# In[15]:


y_pred


# In[16]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[17]:


df


# In[18]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[19]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import SVC
from sklearn import linear_model


# In[20]:


clf = [MultinomialNB(alpha=10), SVC(kernel = 'linear', C=1.5, probability=True), LogisticRegression()]

labels = [ 'Naive Bayes', 'SVM', 'Log regres']

mean_scores = []
cms = []

for i in range(0,3):

    clf[i].fit(X_train,y_train)

    scores = cross_val_score(clf[i], X_train, y_train, cv=10)
    print (labels[i]," : ", scores.mean(),)
    
    mean_scores.append(scores.mean())  


# In[21]:


pip install statsmodels 


# In[22]:


from scipy.stats import poisson,skellam


# In[23]:


df_nfl


# In[24]:


df = df_nfl.loc[df_nfl['Game Location'] != "AWAY"]


# In[25]:


df


# In[26]:


skellam.pmf(0.0,  df['Team1_Score'].mean(),  df['Team2_Score'].mean())


# In[27]:


skellam.pmf(1,  df['Team1_Score'].mean(),  df['Team2_Score'].mean())


# In[28]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[29]:


poisson_model = smf.glm(formula="Team1_Score ~ Team1 + Team2 + Team1_Passing + Team1_Rushing + Team1_Turnovers + Team2_Passing + Team2_Rushing + Team2_Turnovers", data=df, 
                        family=sm.families.Poisson()).fit()
poisson_model.summary()


# In[30]:


def average_performance(teamName):
    df1 = df.loc[df['Team1'] == teamName]
    df2 = df.loc[df['Team2'] == teamName]
    df1 = df1[['Team1_Passing', 'Team1_Rushing', 'Team1_Turnovers']]
    df2 = df2[['Team2_Passing', 'Team2_Rushing', 'Team2_Turnovers']]
    result = pd.concat([df1,df2.rename(columns={'Team2_Passing':'Team1_Passing', 'Team2_Rushing':'Team1_Rushing', 'Team2_Turnovers':'Team1_Turnovers' })], ignore_index=True)
    return(result.mean())
        
    #calculate the team's average passing, rushing, and turnovers


# In[31]:


average_performance('New York Giants')


# In[32]:


def simulate_match(foot_model, t1, t2, max_score = 100):
    team1_stats = average_performance(t1)
    team2_stats = average_performance(t2)
    team1_score_avg = foot_model.predict(pd.DataFrame(data={'Team1': t1, 'Team2': t2, 'Team1_Passing':team1_stats[0], 'Team1_Rushing':team1_stats[1], 'Team1_Turnovers':team1_stats[2],
                                                            'Team2_Passing':team2_stats[0], 'Team2_Rushing':team2_stats[1], 'Team2_Turnovers':team2_stats[2]},
                                                            index=[1])).values[0]
    team2_score_avg = foot_model.predict(pd.DataFrame(data={'Team1': t2, 'Team2': t1, 'Team2_Passing':team1_stats[0], 'Team2_Rushing':team1_stats[1], 'Team2_Turnovers':team1_stats[2],
                                                            'Team1_Passing':team2_stats[0], 'Team1_Rushing':team2_stats[1], 'Team1_Turnovers':team2_stats[2]},
                                                            index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_score+1)] for team_avg in [team1_score_avg, team2_score_avg]]
    return[(np.outer(np.array(team_pred[0]), np.array(team_pred[1]))), team1_score_avg, team2_score_avg]


# In[33]:


match = simulate_match(poisson_model, 'New York Giants', 'Arizona Cardinals', max_score = 60)
#NY Giants win
np.sum(np.tril(match[0], -1))


# In[34]:


#Draw
np.sum(np.diag(match[0]))


# In[35]:


#Arizona Cardinals win
np.sum(np.triu(match[0], 1))


# In[36]:


match[1]


# In[37]:


match[2]


# In[38]:


match2 = simulate_match(poisson_model, 'San Francisco 49ers', 'Green Bay Packers', max_score = 60)
#SF 49ers win
np.sum(np.tril(match2[0], -1))


# In[39]:


#Draw
np.sum(np.diag(match2[0]))


# In[40]:


#Green Bay Packers
np.sum(np.triu(match2[0], 1))


# In[41]:


#match2[1]


# In[42]:


#match2[2]


# In[43]:


match3 = simulate_match(poisson_model, 'Kansas City Chiefs', 'Tennessee Titans', max_score = 100)
#Kansas City win
np.sum(np.tril(match3[0], -1))


# In[44]:


#match3[1]


# In[45]:


#match3[2]


# In[46]:


match4 = simulate_match(poisson_model, 'Green Bay Packers', 'Seattle Seahawks', max_score = 100)
#Green Bay Packers win
np.sum(np.tril(match4[0], -1))


# In[47]:


np.sum(np.diag(match4[0]))


# In[48]:


np.sum(np.triu(match4[0], 1))


# In[49]:


match5 = simulate_match(poisson_model, 'Kansas City Chiefs', 'Houston Texans', max_score = 100)
#Kansas City Chiefs win
np.sum(np.tril(match5[0], -1))


# In[54]:


match6 = simulate_match(poisson_model, 'San Francisco 49ers', 'Kansas City Chiefs', max_score = 100) 
#San Francisco 49ers win
np.sum(np.tril(match6[0], -1))


# In[55]:


#Kansas City Chiefs win
np.sum(np.tril(match6[0], 1))


# In[52]:


match6[1]


# In[53]:


match6[2]


# In[56]:


df


# In[88]:


import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[57]:


import statsmodels.api as sm


# In[59]:


target = df["Team1_Score"]


# In[62]:


import statsmodels.api as sm

X = df[["Team2_Score", "Team1_Passing", "Team1_Rushing", "Team1_Turnovers", "Team2_Passing", "Team2_Rushing", "Team2_Turnovers"]]
y = target

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[102]:


y_true = df['Team1_Score']
y_pred = predictions


# In[103]:


print(metrics.mean_absolute_error(y_true, y_pred))
print(metrics.mean_squared_error(y_true, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))


# Team1_Score = 0.1934*Team2_Score + 0.0625*Team1_Passing + 0.0768*Team1_Rushing - 2.4253*Team1_Turnovers - 0.0106*Team2_Passing - 0.0172 * Team2_Rushing + 2.3441*Team2_Turnovers

# In[69]:


def avg_performance(teamName):
    df1 = df.loc[df['Team1'] == teamName]
    df2 = df.loc[df['Team2'] == teamName]
    df1 = df1[['Team1_Passing', 'Team1_Rushing', 'Team1_Turnovers', 'Team1_Score']]
    df2 = df2[['Team2_Passing', 'Team2_Rushing', 'Team2_Turnovers', 'Team2_Score']]
    result = pd.concat([df1,df2.rename(columns={'Team2_Passing':'Team1_Passing', 'Team2_Rushing':'Team1_Rushing', 'Team2_Turnovers':'Team1_Turnovers', 'Team2_Score':'Team1_Score'})], ignore_index=True)
    return(result.mean())


# In[70]:


avg_performance("San Francisco 49ers")


# In[71]:


avg_performance("Kansas City Chiefs")


# In[72]:


predicted_49ers = 0.1934*28.1875 + 0.0625*237.0000 + 0.0768*144.0625 - 2.4253*1.4375 - 0.0106*281.1250 - 0.0172 * 98.0625 + 2.3441*0.9375


# In[73]:


predicted_49ers


# In[74]:


predicted_chiefs = 0.1934*29.9375 + 0.0625*281.1250 + 0.0768*98.0625 - 2.4253*0.9375 - 0.0106*237.0000 - 0.0172 * 144.0625 + 2.3441*1.4375


# In[75]:


predicted_chiefs


# In[131]:


import statsmodels.api as sm

X2 = df[["Team2_Score", "Team1_Passing", "Team1_Rushing", "Team1_Turnovers", "Team2_Rushing", "Team2_Turnovers"]]
y2 = target

# Note the difference in argument order
model2 = sm.OLS(y2, X2).fit()
predictions2 = model2.predict(X2) # make the predictions by the model

# Print out the statistics
model2.summary()


# In[137]:


y_pred2 = predictions2
print(metrics.mean_absolute_error(y_true, y_pred2))
print(metrics.mean_squared_error(y_true, y_pred2))
print(np.sqrt(metrics.mean_squared_error(y_true, y_pred2)))


# In[132]:


predicted2_49ers = 0.12 * 28.1875 + 0.0598 * 237.0000 + 0.0729 * 144.0625 - 2.2379 * 1.4375 - 0.0146 * 98.0625 + 2.1772 * 0.9375


# In[133]:


predicted2_49ers


# In[134]:


predicted2_chiefs = 0.12 * 29.9375 + 0.0598 * 281.1250 + 0.0729 * 98.0625 - 2.2379 * 0.9375 - 0.0146 * 144.0625 + 2.1772 *  1.4375


# In[135]:


predicted2_chiefs


# In[ ]:




