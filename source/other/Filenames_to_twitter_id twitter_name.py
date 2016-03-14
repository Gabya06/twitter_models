
# coding: utf-8

# In[85]:

import boto
from twitter.models import TwitterUser
from mappings.models import UserMapping
from lib.utils import parse_dash_separated_date
import pandas as pd
import numpy as np
import io


# In[122]:

S3_CONNECTION = boto.connect_s3(settings.AMAZON_S3_KEY,
                                settings.AMAZON_S3_SECRET)
S_BUCKET = S3_CONNECTION.get_bucket('shareablee')

bucket_name = 'shareablee' # for getting data
bucket_tmp = 'shareablee-hive' # for writing data

prefix = 'tmp/twitter/'
write_key_pattern = '_DONE'

TWITTER = 'twitter'

SERVICES = [TWITTER]

SERVICE_DICT = dict(
   [
       (
           service,
           UserMapping.objects.all().prefetch_related(service)
       )
       for service in SERVICES
   ]
)

START_DATE = '2016-01-01'
END_DATE = '2016-01-31'

start_date = parse_dash_separated_date(START_DATE)
end_date = parse_dash_separated_date(END_DATE)


# In[103]:

twitter_id = flagged_prop
tmp_list = SERVICE_DICT.get(TWITTER).filter(twitter__user_id__in = twitter_id, twitter__isnull=False)


# In[110]:

tmp_list.values_list('twitter__user_id', flat = True)
tmp_list.values()


# In[2]:

def connect_to_s3():
    """ Return a connection to Shareablee's S3 resource
    """
    key = settings.AMAZON_S3_KEY
    secret = settings.AMAZON_S3_SECRET
    return boto.connect_s3(key, secret)


# In[3]:

pd_kwargs = {
    'header': 1,
    'sep': ',',
    'engine': 'c',
    'header': 0,
    'escapechar': '\\',
    'usecols' : ['time', 'impressions','retweets']
}



# In[4]:

conn = connect_to_s3()
key = "/".join(['twitter_analytics', '1.1', 'analytics_reports/']) 
res = list(S_BUCKET.list(prefix=key)) # list of files


# In[80]:


def get_prop_id(filename):
    """ Returns property_id from filename
    """
    return filename.key.split("/")[-1].split('_')[0]


# def get_keys(conn, bucket_name, key_pattern, **kwargs):
#     """ Returns an in-memory pandas database by matching s3 key names to a regex pattern
#     """
#     bucket = conn.get_bucket(bucket_name)
#     prefix = kwargs.get('prefix', '')
    
#     results = pd.DataFrame(columns=pd_kwargs['names'])
    
#     for k in bucket.list(prefix=prefix, delimiter="/"):
#         matched = re.search(re.compile(key_pattern), k.name)
#         if matched:
#             print 'matched!', k.name
#             data = pd.read_csv(k, **pd_kwargs)
#             results = results.append(data)

#     print results.sort(columns=['users'], ascending=False).head()
    
#     return results

def write_key(data, conn, bucket_name, **kwargs):
    """ Writes data from a pandas dataframe to an S3 key
    """
    bucket = conn.get_bucket(bucket_tmp)
    key = boto.s3.key.Key(bucket)
    output_prefix = kwargs.get('output_prefix', '')

    tmp_file = io.BytesIO()
    data.to_csv(tmp_file, encoding='utf-8')
    tmp_file.seek(0)
    key.key = output_prefix
    key.set_contents_from_file(tmp_file)


# In[ ]:




# In[ ]:




# In[6]:

flagged_prop = ['30309979', '18342955', '16374678','119606058','16560657', '226299107',
'9695312', '32448740', '1426645165', '21308602','759251','73200694','634784951','14946736','27677483',
'25053299','436171805','25453312','14934818','192981351','2367911','19426551','25589776','223525053',
                '5988062','14293310','40924038','15513910']


# In[7]:

file_names = [i.key.split("/")[-1] for i in res if get_prop_id(i) in flagged_prop]

filtered_names = [i for i in  file_names if i.split('_')[2].split("-")[0] =='2016' 
                  if i.split('_')[2].split("-")[1] =='01' or i.split('_')[2].split("-")[1] =='02']


# In[22]:

tmp = []
def filter_weeks(x):
    """ Returns list of files with the below timestamps
    """
    start_week = x.split("_")[2]
    end_week = x.split("_")[3]
    week_1 = ['2016-01-01', '2016-01-08']
    week_2 = ['2016-01-09', '2016-01-16']
    week_3 = ['2016-01-17', '2016-01-24']
    week_4 = ['2016-01-26', '2016-02-02']
    date_part = x.split('_')[2:4]
    if date_part in [week_1, week_2, week_3, week_4]:
        tmp.append(x)
    else:
        pass
    return tmp


# In[142]:

tw_fans_df = pd.DataFrame(columns=['twitter_id', 'fans'])
def get_tw_fans(user_pk, **kwargs):
    tw_ids = []
    tw_fans = []
#    user = SERVICE_DICT.get(TWITTER).filter(twitter__pk=user_pk, twitter__isnull=False).first().twitter
    tmp_list = SERVICE_DICT.get(TWITTER).filter(twitter__user_id__in = twitter_id, twitter__isnull=False)
    for user in tmp_list:
        user.twitter.set_date_range(start_date, end_date) 
        stats = user.twitter.stats()
        fans = stats.get('latest_followers', 0)
        tw_ids.append(user.twitter.user_id)
        tw_fans.append(fans)
    tw_fans_df['twitter_id'] = tw_ids
    tw_fans_df['fans'] = tw_fans
    return tw_fans_df


# In[150]:

# get twitter follower count (fans) 
tw_fans_df = get_tw_fans(flagged_prop)
tw_fans_df.head()


# In[159]:

def write_files():
    """ Returns dataframe with: time, impressions, retweets, fans and page_id for Jan 2016
        Also write each file to tmp/twitter & all_data to file
    """
    all_data = pd.DataFrame()
    for k in res:
        if k.name.split("/")[-1] in tmp:
            data = pd.read_csv(k, **pd_kwargs)
            page_id = k.key.split("/")[-1].split("_")[0]
            data['page_id'] = page_id
            write_key(data, conn, bucket_tmp, **{'output_prefix': prefix + page_id + write_key_pattern})
            print page_id, 'Done!', '\n'
            all_data = all_data.append(data)
    # merge results with twitter follower count
    all_data = pd.merge(all_data, tw_fans_df, left_on = 'page_id', right_on='twitter_id', how ='left' )
    # write all data in one file
    write_key(all_data, conn, bucket_tmp, **{'output_prefix': prefix + "ALL_DATA" + write_key_pattern})
    print "DONE WRITING ALL DATA"
    return all_data

all_data = write_files()


# In[ ]:

# put results into list - only want twitter_id
user_id_list = np.unique([i.key.split("/")[-1].split('_')[0] for i in res]).tolist()


# In[ ]:

# list of user_id, mapping_id, and user_name
user_id_list = TwitterUser.objects.filter(user_id__in=user_id_list).values_list('user_id',flat=True)
mapping_id_list = TwitterUser.objects.filter(user_id__in=user_id_list).values_list('id',flat=True)
user_name_list = TwitterUser.objects.filter(user_id__in=user_id_list).values_list('username',flat=True)


# In[ ]:

id_names_dict = dict(zip(user_id_list, user_name_list))


# In[ ]:

get_ipython().magic(u"store id_names_dict  >> 'names.txt'")


# In[ ]:

dict(zip( user_id_list, zip(mapping_id_list, user_name_list)))


# In[ ]:



