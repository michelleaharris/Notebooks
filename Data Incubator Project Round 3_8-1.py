
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


# In[360]:

seaborn.set(style="darkgrid", color_codes=True)

gf = seaborn.jointplot("AVERAGE_BUILDING_AGE", "TOTAL_KWH", data=df_sf3, kind="reg", xlim=(0, 110), ylim=(0, 250000), color="r")

plt.xlabel("Average Building Age")
plt.ylabel("kWh")
plt.show(gf)


# In[362]:

seaborn.set(style="darkgrid", color_codes=True)

gf1 = seaborn.jointplot("AVERAGE_HOUSESIZE", "TOTAL_KWH", data=df_sf3, kind="reg", xlim=(1.0, 5.0), ylim=(0, 250000), color="r")

plt.xlabel("Average Building Size")
plt.ylabel("kWh")
plt.show(gf1)


# In[364]:

seaborn.set(style="darkgrid", color_codes=True)

gf2 = seaborn.jointplot("AVERAGE_STORIES", "TOTAL_KWH", data=df_sf3, kind="reg", xlim=(1.0, 2.0), ylim=(0, 250000), color="r")

plt.xlabel("Average Building Stories")
plt.ylabel("kWh")
plt.show(gf2)


# In[367]:

seaborn.set(style="darkgrid", color_codes=True)

gf3 = seaborn.jointplot("AVERAGE_BUILDING_AGE", "TOTAL_THERMS", data=df_sf3, kind="reg", xlim=(0, 110), ylim=(0, 30000), color="b")

plt.xlabel("Average Building Age")
plt.ylabel("Therms")
plt.show(gf3)


# In[368]:

seaborn.set(style="darkgrid", color_codes=True)

gf4 = seaborn.jointplot("AVERAGE_HOUSESIZE", "TOTAL_THERMS", data=df_sf3, kind="reg", xlim=(1.0, 5.0), ylim=(0, 30000), color="b")

plt.xlabel("Average Building Size")
plt.ylabel("Therms")
plt.show(gf4)


# In[369]:

seaborn.set(style="darkgrid", color_codes=True)

gf5 = seaborn.jointplot("AVERAGE_STORIES", "TOTAL_THERMS", data=df_sf3, kind="reg", xlim=(1.0, 2.0), ylim=(0, 30000), color="b")

plt.xlabel("Average Building Stories")
plt.ylabel("Therms")
plt.show(gf5)


# In[387]:

dfg_nyc=pd.read_csv('Gas_NYC.csv')
dfg_nyc.columns = pd.Series(dfg_nyc.columns).str.replace(' ','')
dfg_nyc.columns = pd.Series(dfg_nyc.columns).str.replace('(','')
dfg_nyc.columns = pd.Series(dfg_nyc.columns).str.replace(')','')


# In[385]:

dfg1=dfg_nyc.ZipCode.str.split('\n').str.get(0)


# In[389]:

dfg_nyc1=pd.concat([dfg1,dfg_nyc.Consumptiontherms],axis=1)


# In[410]:

dfg_nyc1.columns = ['JURISDICTION_NAME', 'Therms']


# In[433]:

dfga_nyc=dfg_nyc1.groupby('JURISDICTION_NAME', as_index=False).mean()


# In[436]:

dfga_nyc1=dfga_nyc.apply(lambda x: pd.to_numeric(x, errors='ignore'))


# In[372]:

dfp_nyc=pd.read_csv('Demographic_NYC.csv')
dfp_nyc.columns = pd.Series(dfp_nyc.columns).str.replace(' ','_')


# In[432]:

dfpa_nyc=dfp_nyc.groupby('JURISDICTION_NAME', as_index=False).mean()


# In[437]:

df_nyc=pd.merge(dfga_nyc1, dfpa_nyc, on='JURISDICTION_NAME', how='inner')


# In[453]:

seaborn.set(style="darkgrid", color_codes=True)

nyc = seaborn.jointplot("PERCENT_RECEIVES_PUBLIC_ASSISTANCE", "Therms", data=df_nyc, kind="reg", 
                        xlim=(-0.1, 1.1), ylim=(-1000000, 10000000), color="#4CB391")

plt.xlabel("Fraction Receives Public Assistance")
plt.ylabel("Therms x10^7")
plt.show(nyc)

