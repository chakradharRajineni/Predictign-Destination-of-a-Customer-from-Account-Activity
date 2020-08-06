#!/usr/bin/env python
# coding: utf-8

# In[1]:


+
import plotly.graph_objs as go


# In[2]:


test_data=pd.read_csv('test_users.csv', parse_dates=['timestamp_first_active','date_account_created','date_first_booking'])
train_data=pd.read_csv('train_users_2.csv', parse_dates=['timestamp_first_active','date_account_created','date_first_booking'])


# We have two datasets, one is for training and one is for testing the models. Therefore, let see the similarities between two data sets.

#Data Exploration

#   Train Data  

# In[3]:


train_data.head()


# In[4]:


train_data.dtypes


# In[5]:


train_data.shape


# In[6]:


train_data.columns


# In[7]:


uniqueValues = train_data.nunique(dropna=False)
 
print('Count of unique values in each column :')
print(uniqueValues)


# In[8]:


uv=train_data[[ 'gender', 'signup_method',
       'signup_flow', 'language', 'affiliate_channel',
       'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
       'first_device_type', 'first_browser', 'country_destination']]
for col in uv:
    print(col,':    ', uv[col].unique())


# In[9]:


train_data.replace('-unknown-', np.nan, inplace=True)
train_data.isnull().sum()/len(train_data)


# In[10]:


import missingno as msno
msno.matrix(train_data)


# Lets Check the Test Data

# In[11]:


test_data.head()


# In[12]:


test_data.columns


# We can see that, country_id is not present in teh test data and therefore, predicting the accuracy is not part of this project.

# In[13]:


test_data.shape


# In[14]:


len(test_data)/len(train_data)


# In[15]:


test_data.replace('-unknown-', np.nan, inplace=True)
test_data.isnull().sum()/len(train_data)


# In[16]:


msno.matrix(test_data)


# From the above matrix graph, we can see that in test data, data_first booking is completely missing. Therefore, for modelling purpose, data_first_booking in train data has no importance. Therefore, dropping the column.

# In[17]:


test_data=test_data.drop('date_first_booking', axis=1)
train_data=train_data.drop('date_first_booking', axis=1)


# # Univariate Exploration

# #### Target colum=Country Destination

# In[18]:


g=sns.catplot(y="country_destination", kind="count", data=train_data, height=5, aspect=2.5, orient='h') 


# NDF is no booking, which is majority. 
# The data being from Newyork, the popular country of booking is US. other being, set of countries not defined, has second highest booking. 

# #### Distribution of Age

# In[19]:


sns.kdeplot(train_data['age'], shade=True, color='blue', kernel='gau')
plt.xlim(0,120)
plt.title('Distibution of Age')


# #age>75 travelling and usinf airbnb doesnot make sense that much
# #age<18 is legally not allowed
# #so lets replace them by Nan

# In[20]:


train_data.loc[train_data.age > 75, 'age'] = np.nan
train_data.loc[train_data.age < 18, 'age'] = np.nan
train_data.isnull().sum()/len(train_data)


# In[21]:


sns.kdeplot(train_data['age'], shade=True, color='blue', kernel='gau')
plt.xlim(0,120)
plt.title('Distibution of Age')


# By the distribution, we can see that people between 20-50 used Airbnb more

# In[22]:


id_col=train_data['id']
tar_col=train_data['country_destination']
trd=train_data.drop(['id','country_destination'],axis=1)
cat_cols=[i for i in trd.columns if trd[i].dtype=='object']
cat_cols


# In[23]:


def histogram(column) :
    trace = go.Histogram(x  = trd[column],
                          histnorm= "percent",
                          name = "Histogram of "+column+ " variables",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    data = [trace]
    layout = go.Layout(dict(title =" Distribution of "+ column,
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = column,zerolinewidth=1,
                                             ticklen=5,gridwidth=2 ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',title = "percent",zerolinewidth=1,
                                             ticklen=5,gridwidth=2 ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)
    
for i in cat_cols:
    histogram(i)


# ## Bivariate Analysis

# #### Age vs Country_destination

# In[24]:


f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='country_destination', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data, sizes=(1, 8), linewidth=0, ax=ax)


# There is no particluar differentiation in country selection,we can define from age variable.

# Age vs signup Method

# In[25]:


f, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='signup_method', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data,linewidth=2, saturation=2,  ax=ax)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)


# We can see that googleis used by lower age people, while 'basic'(directly in the site) is used by older people.

# In[26]:


f, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='signup_app', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data,linewidth=2, saturation=2,  ax=ax)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# 

# In[27]:


f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='language', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data,linewidth=2, saturation=2,  ax=ax)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# 

# In[28]:


f, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='affiliate_channel', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data,linewidth=2, saturation=2,  ax=ax)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# 

# In[29]:


f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='affiliate_provider', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data,linewidth=2, saturation=2,  ax=ax)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# 

# In[30]:


f, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='first_affiliate_tracked', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data,linewidth=2, saturation=2,  ax=ax)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


# 

# In[31]:


f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='first_device_type', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data,linewidth=2, saturation=2,  ax=ax)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# 

# In[32]:


f, ax = plt.subplots(figsize=(30, 7))
sns.boxplot(x='first_browser', y='age',
                palette="ch:r=-.2,d=.3_r", data=train_data,linewidth=2, saturation=2,  ax=ax)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


# In[33]:


f, ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='date_account_created', y='age', data=train_data)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# In[34]:


trd['dac_year'] = trd.date_account_created.dt.year
trd['dac_month'] = trd.date_account_created.dt.month
trd['dac_day'] = trd.date_account_created.dt.weekday_name


# In[35]:


f, ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='dac_year', y='age', data=trd)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# In[36]:


f, ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='dac_month', y='age', data=trd)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# In[37]:


f, ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='dac_day', y='age', data=trd)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# In[38]:


f, ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='timestamp_first_active', y='age', data=trd)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# In[39]:


trd['dfa_year'] = trd.timestamp_first_active.dt.year
trd['dfa_month'] = trd.timestamp_first_active.dt.month
trd['dfa_day'] = trd.timestamp_first_active.dt.weekday_name


# In[40]:


f, ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='dfa_year', y='age', data=trd)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# In[41]:


f, ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='dfa_month', y='age', data=trd)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# In[42]:


f, ax = plt.subplots(figsize=(30, 10))
sns.lineplot(x='dfa_day', y='age', data=trd)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)


# In[43]:


trd.columns.values


# #### Gender

# In[44]:


train_data.gender.value_counts()


# In[45]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='country_destination', data=train_data, height=10, aspect=2, orient='h', legend=False) 
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xlabel('country_destination', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend( prop={'size': 30})
sns.despine()


# We can see that data has more Female population that Male population and is reflected in destination.
# #one reflection we can clearly see is that, male have booked to other countries more than female countries.

# 

# In[46]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='signup_method', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup Method', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[47]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='signup_flow', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup Flow', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[48]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='language', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Language', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(0,1000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[49]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='affiliate_channel', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Affiliate channel', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[50]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='affiliate_provider', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Affiliate Provider', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[51]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='first_affiliate_tracked', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Affiliate Provider', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[52]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='signup_app', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('SignUp App', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[53]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='first_device_type', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First device type', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[54]:


print(len(train_data[train_data['gender']=='FEMALE'])/len(train_data[train_data['gender']=='MALE']))
sns.catplot(hue="gender", kind="count",x='first_browser', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Browser', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(0,5)
plt.legend( prop={'size': 30})
sns.despine()


# #### Sign up Method

# 

# In[55]:


sns.catplot(hue="signup_method", kind="count",x='signup_flow', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup Flow', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(0,10000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[56]:


sns.catplot(hue="signup_method", kind="count",x='language', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Language', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(0,2000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[57]:


sns.catplot(hue="signup_method", kind="count",x='affiliate_channel', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Affliate Channel', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim(0,10000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[58]:


sns.catplot(hue="signup_method", kind="count",x='affiliate_provider', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Affliate Provider', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim(0,10000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[59]:


sns.catplot(hue="signup_method", kind="count",x='first_affiliate_tracked', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Affiliate Tracked', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim(0,10000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[60]:


sns.catplot(hue="signup_method", kind="count",x='signup_app', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup App', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim(0,10000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[61]:


sns.catplot(hue="signup_method", kind="count",x='first_device_type', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Device Type', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim(0,10000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[62]:


sns.catplot(hue="signup_method", kind="count",x='first_browser', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Browser', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,10)
plt.legend( prop={'size': 30})
sns.despine()


# #### Signup Flow

# 

# In[63]:


sns.catplot(hue="signup_flow", kind="count",x='language', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Language', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,20000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[64]:


sns.catplot(hue="signup_flow", kind="count",x='affiliate_channel', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Affliate Channel', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,20000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[65]:


sns.catplot(hue="signup_flow", kind="count",x='affiliate_provider', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Affliate Provider', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,20000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[66]:


sns.catplot(hue="signup_flow", kind="count",x='first_affiliate_tracked', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Affiliate Tracked', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,20000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[67]:


sns.catplot(hue="signup_flow", kind="count",x='signup_app', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup App', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,20000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[68]:


sns.catplot(hue="signup_flow", kind="count",x='first_device_type', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Device Type', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,20000)
plt.legend( prop={'size': 30})
sns.despine()


# 

# In[69]:


sns.catplot(hue="signup_flow", kind="count",x='first_browser', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Browser', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,20000)
plt.legend(loc=4, prop={'size': 30})
sns.despine()


# 

# #### Affliate Channel

# 

# In[70]:


sns.catplot(hue="affiliate_channel", kind="count",x='affiliate_provider', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Affiliate Provider', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
plt.ylim(0,30000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# In[71]:


sns.catplot(hue="affiliate_channel", kind="count",x='first_affiliate_tracked', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Affiliate Tracked', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
#plt.ylim(0,30000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# In[72]:


sns.catplot(hue="affiliate_channel", kind="count",x='signup_app', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup App', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
plt.ylim(0,30000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# In[73]:


sns.catplot(hue="affiliate_channel", kind="count",x='first_device_type', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Device Type', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
plt.ylim(0,30000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# In[74]:


sns.catplot(hue="affiliate_channel", kind="count",x='first_browser', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Browser', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,30000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# #### Alliate Provider

# 

# In[75]:


sns.catplot(hue="affiliate_provider", kind="count",x='first_affiliate_tracked', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Affiliate Tracked', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
plt.ylim(0,10000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# In[76]:


sns.catplot(hue="affiliate_provider", kind="count",x='signup_app', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup App', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
plt.ylim(0,20000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# In[77]:


sns.catplot(hue="affiliate_provider", kind="count",x='signup_app', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup App', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
plt.ylim(0,30000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# In[78]:


sns.catplot(hue="affiliate_provider", kind="count",x='first_device_type', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Device Type', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
#plt.ylim(0,10000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# In[79]:


sns.catplot(hue="affiliate_provider", kind="count",x='first_browser', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('First Browser', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
plt.ylim(0,10000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# In[80]:


sns.catplot(hue="first_affiliate_tracked", kind="count",x='signup_app', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('Signup App', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
#plt.ylim(0,10000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# In[81]:


sns.catplot(hue="first_affiliate_tracked", kind="count",x='first_device_type', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('first_device_type', fontsize=40, fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.xlim(-1,5)
#plt.ylim(0,10000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# In[82]:


sns.catplot(hue="first_affiliate_tracked", kind="count",x='first_browser', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('first browser',fontsize=40,fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
#plt.ylim(0,10000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# In[83]:


sns.catplot(hue="signup_app", kind="count",x='first_browser', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('first browser',fontsize=40,fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
#plt.ylim(0,10000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# In[84]:


sns.catplot(hue="first_affiliate_tracked", kind="count",x='first_device_type', data=train_data, height=10, aspect=3, orient='h', legend=False) 
plt.xlabel('first Device Type',fontsize=40,fontweight='bold')
plt.ylabel('Count', fontsize=40, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim(-1,5)
#plt.ylim(0,10000)
plt.legend( loc=4,prop={'size': 30})
sns.despine()


# 

# 

# #### Data Account Created

# ### Feature Engineering

# In[85]:


train_data.columns


# In[86]:


id_col=train_data['id']
tar_col=train_data['country_destination']
final_df=train_data.drop(['id','country_destination'], axis=1)


# In[87]:


final_df['dac_year'] = final_df.date_account_created.dt.year
final_df['dac_month'] = final_df.date_account_created.dt.month
final_df['dac_day'] = final_df.date_account_created.dt.weekday_name
final_df['tac_year'] = final_df.timestamp_first_active.dt.year
final_df['tac_month'] = final_df.timestamp_first_active.dt.month
final_df['tac_day'] = final_df.timestamp_first_active.dt.weekday_name


# In[88]:


final_df=final_df.drop(['date_account_created','timestamp_first_active'], axis=1)


# In[89]:


final_df.columns


# In[90]:


final_df.dtypes


# In[91]:


final_df.isnull().sum()


# ### Missing Values

# Lets replace the missing values with 'unknown' as new category in categorical variable columns

# Age

# In[92]:


mean_age=round(final_df.age.mean(),0)
final_df.age.fillna(mean_age,axis=0, inplace=True)


# In[93]:


final_df.age.isnull().sum()


# ##### Gender

# In[94]:


final_df.gender.unique()


# In[95]:


final_df.gender.replace(np.nan, 'unknown', inplace=True)


# In[96]:


final_df.gender.value_counts()


# As there are only 4 varables, we need not decrease the number of categories.

# ##### Signup Method

# In[97]:


final_df.signup_method.value_counts()


# Only three categories, therefore neednot eliminate any categories

# Signup Flow

# In[98]:


final_df.signup_flow.value_counts()


# In[99]:


final_df['signup_flow'].replace([6,8,21,5,20,16,15,10,4], '99', inplace=True)


# All the categories with less than 0.1% of total are converted into seperate category as 99

# ##### Language

# In[100]:


final_df.language.value_counts()


# In[101]:


final_df['language'].replace(['ja','sv','nl','tr','da','pl','cs','no','el','th','id','hu','fi','is','ca','hr'], 'x', inplace=True)


# In[102]:


final_df.language.value_counts()


# ##### Affliate Channel

# In[103]:


final_df.affiliate_channel.value_counts()


# No need to replace any Affliate Channel, as already they are put in other category

# In[104]:


final_df.affiliate_provider.value_counts()


# In[105]:


final_df['affiliate_provider'].replace(['email-marketing','naver','baidu','yandex','wayn','daum'], 'other', inplace=True)


# In[106]:


final_df.affiliate_provider.value_counts()


# All the categories that have less than 200 inputs are put into other category

# ##### First Affliate Tracked

# In[107]:


final_df.first_affiliate_tracked.replace(np.nan, 'unknown')


# In[108]:


final_df.first_affiliate_tracked.value_counts()


# Let's put marketing and local ops into other category

# In[109]:


final_df['first_affiliate_tracked'].replace(['marketing', 'local ops'], 'other', inplace=True)


# In[110]:


final_df.first_affiliate_tracked.value_counts()


# ##### Signup App

# In[111]:


final_df.signup_app.value_counts()


# No need to replace any category here

# ##### First Device Type

# In[112]:


final_df.first_device_type.value_counts()


# No need to replace any category

# ##### First browser

# In[113]:


final_df.first_browser.replace(np.nan, 'unknown')


# In[114]:


x=[[final_df.first_browser.value_counts()<=200]]
x


# In[115]:


final_df['first_browser'].replace(['Outlook 2007','Arora', 'Epic','Opera','Silk','Chromium','BlackBerry Browser','Maxthon','Apple Mail',                 
'IE Mobile','Sogou Explorer','Mobile Firefox','SiteKiosk','RockMelt','Iron','IceWeasel','Pale Moon', 'CometBird','Yandex.Browser','SeaMonkey','Camino','TenFourFox',                 
'wOSBrowser','CoolNovo','Opera Mini','Avant Browser','Mozilla', 'Flock','TheWorld Browser','Comodo Dragon','Opera Mobile',               
'Crazy Browser','OmniWeb','SlimBrowser','Kindle Browser','Stainless','Googlebot','Google Earth','PS Vita browser',            
'IceDragon','Conkeror','NetNewsWire','Palm Pre web browser'], 'other', inplace=True)


# In[116]:


final_df.first_browser.value_counts()


# In[117]:


final_df.columns.values


# ### Encoding

# In[118]:


final_df['dac_year']=pd.Categorical(final_df.dac_year)
final_df['dac_month']=pd.Categorical(final_df.dac_month)
final_df['tac_year']=pd.Categorical(final_df.tac_year)
final_df['tac_month']=pd.Categorical(final_df.tac_month)
final_df.dtypes


# In[119]:


final_df= pd.get_dummies(final_df, columns= [i for i in final_df.columns if final_df[i].dtypes=='object'],drop_first=True)


# In[120]:


final_df.head()


# In[121]:


dac_year_dummies=pd.get_dummies(final_df.dac_year,prefix='dac_year').iloc[:, 1:]


# In[122]:


pd.concat([final_df, dac_year_dummies], axis=1)


# In[123]:


final_df=final_df.drop('dac_year', axis=1)


# In[124]:


dac_month_dummies=pd.get_dummies(final_df.dac_month,prefix='dac_month').iloc[:, 1:]
tac_year_dummies=pd.get_dummies(final_df.tac_year,prefix='tac_year').iloc[:, 1:]
tac_month_dummies=pd.get_dummies(final_df.tac_month,prefix='tac_month').iloc[:, 1:]


# In[125]:


pd.concat([final_df, dac_month_dummies], axis=1)
pd.concat([final_df, tac_year_dummies], axis=1)
pd.concat([final_df, tac_month_dummies], axis=1)


# In[126]:


final_df=final_df.drop(['dac_month','tac_year','tac_month'], axis=1)


# In[127]:


final_df.columns.values


# ### Sampling Data set into train and Validation sets

# In[162]:


trgt_col=train_data['country_destination']

from sklearn.model_selection import train_test_split
pred_train, pred_test, trgt_train, trgt_test=train_test_split(final_df, trgt_col, test_size=0.2)


# In[163]:


pred_train.shape


# In[164]:


trgt_train.shape


# ### Data Normalization

# In[165]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
pred_train = sc.fit_transform(pred_train)
pred_train=pd.DataFrame(pred_train,columns=predictors.columns)
pred_test=sc.transform(pred_test)



#                      
#                      

# #### Logistic Regression

# In[167]:


final_df.shape



# In[152]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve

#Model
lr=LogisticRegression()
result=lr.fit(pred_train,trgt_train)
pred_lr=lr.predict(pred_test)
print(metrics.accuracy_score(trgt_test,pred_lr))


# In[153]:


print(classification_report(trgt_test, pred_lr))


# In[188]:


#Confusion Matrix with heatmap
cnfsn_matrix=confusion_matrix( trgt_test ,pred_lr)
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cnfsn_matrix, xticklabels=['AU','CA','DE','ES','FR','GB','IT','NDF','NL','PT','US','other']
            , yticklabels=['AU','CA','DE','ES','FR','GB','IT','NDF','NL','PT','US','other'],annot=True,fmt="d",cbar=False,ax=ax)
plt.ylabel('True Lable')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix")
#fig.set_size_inches(18.5, 10.5)


# In[189]:


weights=pd.Series(lr.coef_[0],index=final_df.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='bar'))


# In[190]:


print(weights.sort_values(ascending=False)[-10:].plot(kind='bar'))






