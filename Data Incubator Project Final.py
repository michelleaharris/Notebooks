
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

dfd=pd.read_csv('Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv')
dfd.columns = pd.Series(dfd.columns).str.replace(' ','_')


# In[3]:

dfd1=dfd[dfd.COMMUNITY_AREA_NAME != 'CHICAGO']


# In[4]:

dfd2=dfd1.sort_values(['COMMUNITY_AREA_NAME'], ascending=True)


# In[5]:

dfe=pd.read_csv('Energy_Usage_2010.csv')
dfe.columns = pd.Series(dfe.columns).str.replace(' ','_')


# In[6]:

dfe1=dfe.groupby('COMMUNITY_AREA_NAME', as_index=False).sum()


# In[7]:

df1=pd.concat([dfe1,dfd2],axis=1)


# In[9]:

from bokeh.charts import Bar, show, output_file

p = Bar(dfe, 'COMMUNITY_AREA_NAME', values='TOTAL_KWH', agg='sum', title="Neighborhood Energy Use 2010", 
        legend=False, bar_width=1, color='#6495ed')
p.width=1200
p.height=500

output_file("bar1.html")
show(p)


# In[10]:

dfs1=pd.concat([dfd2.COMMUNITY_AREA_NAME, dfd2.PER_CAPITA_INCOME_, dfe1.KWH_SQFT_MEAN_2010], axis=1)


# In[11]:

dfs1.columns = ['Neighborhood', 'Per_Capita_Income', 'Energy_Per_SqFt']


# In[12]:

dfe2=dfe[dfe.BUILDING_TYPE != 'Commercial']


# In[13]:

dfe3=dfe2.groupby('COMMUNITY_AREA_NAME', as_index=False).sum()


# In[14]:

dfp=(dfe3.TOTAL_KWH / dfe3.TOTAL_POPULATION)


# In[15]:

dfp1=pd.concat([dfs1, dfp], axis=1)


# In[16]:

dfp1.columns = ['Neighborhood', 'Per_Capita_Income','Energy_Per_SqFt', 'Energy_Use_Per_Person']


# In[19]:

from bokeh.charts import Scatter
from bokeh.io import output_file, show
from bokeh.layouts import row

output_file("twoscatter.html")

s1 = Scatter(dfs1, x='Per_Capita_Income', y='Energy_Per_SqFt', title="Income vs. Energy Use",

            xlabel="Per Capita Income", ylabel="Energy Use (kWh) Per SqFt", color='#8B0000')

s2 = Scatter(dfp1, x='Per_Capita_Income', y='Energy_Use_Per_Person', title="Income vs. Househould Energy Use",

            xlabel="Per Capita Income", ylabel="Energy Use Per Residential Person", color='#8B0000')

show(row(s1, s2))


# In[ ]:



