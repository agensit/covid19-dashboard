import pandas as pd
import json

# Load the data
url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
# url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

df_confirmed = pd.read_csv(url_confirmed, error_bad_lines=False)
df_death = pd.read_csv(url_death, error_bad_lines=False)
# df_recovered = pd.read_csv(url_recovered, error_bad_lines=False)

# Fill the NaN
df_confirmed['Province/State'].fillna(df_confirmed['Country/Region'], inplace=True)
df_death['Province/State'].fillna(df_death['Country/Region'], inplace=True)
# df_recovered['Province/State'].fillna(df_recovered['Country/Region'], inplace=True)

# Reshape
def reshape_df(df, state = 'Confirmed'):
    df.set_index(['Province/State','Country/Region','Lat','Long'], inplace=True)
    df = df.stack().reset_index(drop=False)
    df.rename(columns={'level_4':'Date',0:state},inplace=True)
    return df

df_confirmed = df_confirmed.pipe(reshape_df,'Confirmed')
df_death = df_death.pipe(reshape_df,'Death')
# df_recovered = df_recovered.pipe(reshape_df,'Recovered')


# Merge
df_covid19 = pd.merge(df_confirmed, df_death)
# df_covid19 = pd.merge(df_covid19,df_recovered, how='left')

# converte "Date" to  date type
df_covid19['Date'] = pd.to_datetime(df_covid19['Date'])

# tranlate the 'State' in French 
with open('/Users/axel/Agensit/covid19/collect_data/english_to_french.txt') as json_file:
    french_countries = json.load(json_file)
df_covid19['State'] = df_covid19['Province/State'].map(lambda x: french_countries[x] if x in french_countries else x)

# drop useless column
df_covid19.drop(columns=['Country/Region', 'Province/State'], inplace=True)

# save 
df_covid19.to_csv("/Users/axel/Agensit/covid19/collect_data/df_covid19.csv", index=False)