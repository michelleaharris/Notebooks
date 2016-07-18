
# coding: utf-8

# In[4]:

import pandas as pd
import numpy as np


# In[15]:

dfd=pd.read_csv('Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv')
dfd.columns = pd.Series(dfd.columns).str.replace(' ','_')



# In[16]:

dfd1=dfd[dfd.COMMUNITY_AREA_NAME != 'CHICAGO']


# In[17]:

dfd2=dfd1.sort_values(['COMMUNITY_AREA_NAME'], ascending=True)


# In[55]:

dfe=pd.read_csv('Energy_Usage_2010.csv')
dfe.columns = pd.Series(dfe.columns).str.replace(' ','_')


# In[62]:

dfe1=dfe.groupby('COMMUNITY_AREA_NAME', as_index=False).sum()


# In[63]:

df1=pd.concat([dfe1,dfd2],axis=1)


# In[103]:

from bokeh.charts import Bar, show, output_file

p = Bar(dfe, 'COMMUNITY_AREA_NAME', values='TOTAL_KWH', agg='sum', title="Neighborhood Energy Use 2010", 
        legend=False, bar_width=1, color='#6495ed')
p.width=1200
p.height=500

output_file("bar1.html")
show(p)


# In[106]:

dfs1=pd.concat([dfd2.COMMUNITY_AREA_NAME, dfd2.PER_CAPITA_INCOME_, dfe1.KWH_SQFT_MEAN_2010], axis=1)


# In[109]:

dfs1.columns = ['Neighborhood', 'Per_Capita_Income', 'Energy_Per_SqFt']


# In[116]:

from bokeh.charts import Scatter, output_file, show

p = Scatter(dfs1, x='Per_Capita_Income', y='Energy_Per_SqFt', title="Income vs. Energy Use",

            xlabel="Per Capita Income", ylabel="Energy Use (kWh) Per SqFt", color='#8B0000')

output_file("scatter1.html")
show(p)


# In[90]:

dfe2=dfe[dfe.BUILDING_TYPE != 'Commercial']


# In[92]:

dfe3=dfe2.groupby('COMMUNITY_AREA_NAME', as_index=False).sum()


# In[111]:

dfp=(dfe3.TOTAL_KWH / dfe3.TOTAL_POPULATION)


# In[112]:

dfp1=pd.concat([dfs1, dfp], axis=1)


# In[113]:

dfp1.columns = ['Neighborhood', 'Per_Capita_Income','Energy_Per_SqFt', 'Energy_Use_Per_Person']


# In[117]:

from bokeh.charts import Scatter, output_file, show

p = Scatter(dfp1, x='Per_Capita_Income', y='Energy_Use_Per_Person', title="Income vs. Househould Energy Use",

            xlabel="Per Capita Income", ylabel="Energy Use Per Residential Person", color='#8B0000')

output_file("scatter2.html")
show(p)


# In[ ]:



