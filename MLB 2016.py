#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('baseball_reference_2016_clean.csv')

teams = ['Baltimore Orioles', 'Boston Red Sox', 'Chicago White Sox', 'Cleveland Indians', 'Detroit Tigers',
        'Houston Astros', 'Kansas City Royals', 'Los Angeles Angels of Anaheim', 'Minnesota Twins', 'New York Yankees',
        'Oakland Athletics', 'Seattle Mariners', 'Tampa Bay Rays', 'Texas Rangers', 'Toronto Blue Jays',
        'Arizona Diamondbacks', 'Atlanta Braves', 'Chicago Cubs', 'Cincinnati Reds', 'Colorado Rockies',
        'Los Angeles Dodgers', 'Miami Marlins', 'Milwaukee Brewers', 'New York Mets', 'Philadelphia Phillies',
        'Pittsburgh Pirates', 'San Diego Padres', 'San Francisco Giants', 'St. Louis Cardinals',
         'Washington Nationals']

def teamid(temp_team):
    team = temp_team[0]
    tid = teams.index(str(team))
    return tid

train['away_team'] = train[['away_team']].apply(teamid, axis=1)
train['home_team'] = train[['home_team']].apply(teamid, axis=1)

train = train.drop(['Unnamed: 0', 'attendance', 'date', 'field_type', 'game_type', 'start_time', 'venue',
            'day_of_week', 'temperature', 'wind_speed', 'wind_direction', 'sky', 'total_runs', 'game_hours_dec',
            'season', 'home_team_loss', 'home_team_outcome', 'away_team_runs', 'home_team_runs'], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(train.drop('home_team_win', axis=1),
                                                   train['home_team_win'],
                                                   test_size=0.1)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))


# In[34]:


train.head()


# In[35]:


sns.heatmap(train.isnull())


# In[ ]:




