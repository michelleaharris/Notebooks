
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


# In[41]:

dfe=pd.read_csv('Energy_Usage_2010.csv')
dfe.columns = pd.Series(dfe.columns).str.replace(' ','_')


# In[123]:

dfe2=dfe.replace("Less than 4", 2)


# In[124]:

dfe3=dfe2.apply(lambda x: pd.to_numeric(x, errors='ignore'))


# In[125]:

dfe1=dfe3.groupby('COMMUNITY_AREA_NAME', as_index=False).sum()


# In[262]:

dfe4=dfe3[dfe3.BUILDING_TYPE == 'Residential']


# In[193]:

dfe_add=dfe4.groupby('COMMUNITY_AREA_NAME', as_index=False).sum()
dfe_avg=dfe4.groupby('COMMUNITY_AREA_NAME', as_index=False).mean()


# In[127]:

from bokeh.charts import Bar, show, output_file

p = Bar(dfe, 'COMMUNITY_AREA_NAME', values='TOTAL_KWH', agg='sum', title="Neighborhood Energy Use 2010", 
        legend=False, bar_width=1, color='#6495ed')
p.width=1200
p.height=500

output_file("bar1.html")
show(p)


# In[128]:

dfs1=pd.concat([dfd2.COMMUNITY_AREA_NAME, dfd2.PER_CAPITA_INCOME_, dfe1.KWH_SQFT_MEAN_2010], axis=1)


# In[129]:

dfs1.columns = ['Neighborhood', 'Per_Capita_Income', 'Energy_Per_SqFt']


# In[163]:

dfp=(dfe_add.TOTAL_KWH / dfe_add.TOTAL_POPULATION)


# In[164]:

dfp1=pd.concat([dfs1, dfp], axis=1)


# In[165]:

dfp1.columns = ['Neighborhood', 'Per_Capita_Income','Energy_Per_SqFt', 'Energy_Use_Per_Person']


# In[167]:

from bokeh.charts import Scatter
from bokeh.io import output_file, show
from bokeh.layouts import row

output_file("twoscatter.html")

s1 = Scatter(dfs1, x='Per_Capita_Income', y='Energy_Per_SqFt', title="Income vs. Energy Use",

            xlabel="Per Capita Income", ylabel="Energy Use (kWh) Per SqFt", color='#8B0000')

s2 = Scatter(dfp1, x='Per_Capita_Income', y='Energy_Use_Per_Person', title="Income vs. Househould Energy Use",

            xlabel="Per Capita Income", ylabel="Energy Use Per Residential Person", color='#8B0000')

show(row(s1, s2))


# In[168]:

from scipy import stats

stats.linregress(dfs1.Per_Capita_Income, dfs1.Energy_Per_SqFt)


# In[169]:

stats.linregress(dfp1.Per_Capita_Income, dfp1.Energy_Use_Per_Person)


# In[170]:

df_avg=pd.concat([dfe_avg,dfd2],axis=1)


# In[171]:

df_avg1=df_avg.drop(df_avg.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,
                                    34,35,36,37,38,39,41,42,43,44,45,46,48,49,50,51,52,53,55,56,57,58,59,60,67,69,71]], axis=1)


# In[172]:

dfa_corr=df_avg1.corr()


# In[205]:

import matplotlib.pyplot as plt
import seaborn

mask = np.zeros_like(dfa_corr)
mask[np.triu_indices_from(mask)] = True

seaborn.heatmap(dfa_corr, cmap='coolwarm', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
 
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
plt.show()


# In[174]:

df_avg2=df_avg1.drop([41,47,48,49],axis=0)


# In[175]:

dfa2_corr=df_avg2.corr()


# In[183]:

import matplotlib.pyplot as plt
import seaborn

mask = np.zeros_like(dfa2_corr)
mask[np.triu_indices_from(mask)] = True

seaborn.heatmap(dfa2_corr, cmap='coolwarm', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
 
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
plt.show()


# In[222]:

from bokeh import mpl
from bokeh.plotting import output_file, show

seaborn.violinplot(x='BUILDING_SUBTYPE',y='TOTAL_THERMS',data=dfe4)

output_file("seaborn_violin_therms.html")

show(mpl.to_bokeh())


# In[219]:

from bokeh import mpl
from bokeh.plotting import output_file, show 

seaborn.violinplot(x='BUILDING_SUBTYPE',y='TOTAL_KWH',data=dfe4)

output_file("seaborn_violin_kwh.html")

show(mpl.to_bokeh())


# In[288]:

dfe_sub=dfe4.groupby('BUILDING_SUBTYPE', as_index=True).mean()


# In[289]:

dfe_subk = dfe_sub[[1,2,3,4,5,6,7,8,9,10,11,12]]
dfe_subt = dfe_sub[[16,17,18,19,20,21,22,23,24,25,26,27]]


# In[298]:

dfe_subk.columns=['January','February','March','April','May','June','July','August',
                  'September','October','November','December']
dfe_subt.columns=['January','February','March','April','May','June','July','August',
                  'September','October','November','December']


# In[299]:

dfe_km = dfe_subk.transpose()
dfe_tm = dfe_subt.transpose()


# In[314]:

area_tm=dfe_tm.plot.area(colormap='Blues')

plt.xlabel("Month")
plt.ylabel("Therms")

plt.show(area_tm)


# In[315]:

area_km=dfe_km.plot.area(colormap='Blues')

plt.xlabel("Month")
plt.ylabel("kWh")

plt.show(area_km)


# In[323]:

dfe_an=dfe4[[3,32,66]]


# In[324]:

dfe_an.dropna(axis=0)


# In[329]:

g = seaborn.factorplot(x='GAS_ACCOUNTS', y='AVERAGE_BUILDING_AGE', hue='BUILDING_SUBTYPE', data=dfe_an[dfe_an.GAS_ACCOUNTS<40],
                   capsize=.2, palette="YlGnBu_d", size=6, aspect=.75)
g.despine(left=True)

plt.show(g)


# In[330]:

dfe_sf=dfe4[dfe4.BUILDING_SUBTYPE == 'Single Family']


# In[337]:

from bokeh.charts import Bar, show, output_file

p1 = Bar(dfe_sf, 'COMMUNITY_AREA_NAME', values='TOTAL_KWH', agg='sum', title="Neighborhood Energy Use 2010", 
        legend=False, bar_width=1, color='#6495ed')
p1.width=1200
p1.height=500

output_file("bar_sf.html")
show(p1)


# In[332]:

dfe_sfa=dfe_sf.groupby('COMMUNITY_AREA_NAME', as_index=False).mean()


# In[333]:

df_sf2=pd.concat([dfe_sfa,dfd2],axis=1)


# In[338]:

df_sf3=df_sf2.drop(df_sf2.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,
                                    34,35,36,37,38,39,41,42,43,44,45,46,48,49,50,51,52,53,55,56,57,58,59,60,67,69,71]], axis=1)


# In[339]:

dfsf_corr=df_sf3.corr()


# In[340]:

mask = np.zeros_like(dfsf_corr)
mask[np.triu_indices_from(mask)] = True

seaborn.heatmap(dfsf_corr, cmap='coolwarm', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
 
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
plt.show()


# In[ ]:



