import os

import pandas as pd
import numpy as np
from dateutil.parser import parse
import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

# config dash & plotly
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP]) # added Bootstrap CSS.
import plotly.io as pio
pio.templates.default = "plotly_white"
config_dash = {'displayModeBar': False}
margin = dict(l=0, r=0, t=0, b=0)

# set the date to french format
import locale
locale.setlocale(locale.LC_TIME, "fr_FR")

# need to be delete
df = pd.read_csv('databasefr.csv')
df.drop(columns=['Country/Region', 'Recovered','Province/State'], inplace=True)

# create the counter
last_date = df['Date'].max()
confirmed_count = df[df['Date'] == last_date]['Confirmed'].sum()
death_count = df[df['Date'] == last_date]['Death'].sum()



# map markers' size
df['marker_Confirmed'] = df['Confirmed'].map(lambda x: x ** 0.4)
df['marker_Death'] = df['Death'].map(lambda x: x ** 0.4)
# create readable number
def millify(n):
    if n > 999:
        if n > 1e6-1:
            return f'{round(n/1e6,1)}M'
        return f'{round(n/1e3,1)}K'
    return n

def pretty_date(str_date, date_format):
    date = parse(str_date)
    return date.strftime(date_format) 

# mapbox token acess
with open('mapbox_token.txt') as f:
    lines = [x.rstrip() for x in f]
mapbox_access_token = lines[0]

# DASH APP
# added Bootstrap CSS.
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# #### TEST
# import itertools 

# def find_pairs(lst):
#     return [(a,b) for a,b in itertools.permutations(lst, 2)]

# def dist(df, c1,c2):
#     dff = df#.set_index('State')
#     diff_lat = dff.loc[c1,'Lat'] - dff.loc[c2,'Lat']
#     diff_long =  dff.loc[c1,'Long'] - dff.loc[c2,'Long']
#     return np.sqrt(diff_lat ** 2 + diff_long ** 2)   

# def get_max_dist(df, country_list):
#     country_pairs = find_pairs(country_list)
#     country_dist = []
#     for pair in country_pairs:
#         country_dist.append(dist(df, pair[0], pair[1]))
#     if len(country_pairs) > 0: 
#         return max(country_dist)
#     else:
#         return 1

# # custom buckets
# bins = [0, 10.3, 33.8, 79.8, 98.3, np.inf]
# groups_names = [5, 4, 3, 2, 1]
# def binning(df, country_list):
#     dist = get_max_dist(df ,country_list)
#     return pd.cut([dist], bins, labels=groups_names).tolist()
# ####

server = app.server

# filters component
filters = dbc.Card([
    html.Div([
        dcc.Slider(
            id='date_slider',
            min=0,
            max=len(df['Date'].unique())-1,
            marks={i: date for i, date in enumerate(
                df['Date'].unique()) if parse(date).day == 1},
            value=len(df['Date'].unique())-1),
        dcc.Dropdown(
            id='country_dropdown',
            # label="Menu",
            options=[{'label': country, 'value': country}
                     for country in df['State'].unique()],
            multi=True,
            value=None,
            placeholder='Réaliser une étude par pays',
            className='country-dropdown-el')
    ], style={'width': '100%', 'height': '100%', 'padding': '.9rem'})
], className='filter-card')

app.layout = html.Div(
    [
        dbc.Container(
            [
                # Headers
                dbc.Card(
                    dbc.Row(
                        [
                            html.Div(
                                children=[
                                    dbc.Col(html.H1(id='my_title'), sm=12, md=8),
                                    dbc.Col(html.H2(id='my_date', className='header-date'), sm=12, md=4)
                                ],
                                className='header d-flex'
                            )
                        ],
                        className='d-flex justify-content-between h-75 align-items-center'
                    ),
                    className=' p-3 my-3 w-100 header-container'
                ),

                #Tabs & Filters
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Tabs(id="tabs", value='Confirmed', 
                                     children=[
                                         dcc.Tab(id='tab_conf', label=f'{confirmed_count}', value='Confirmed', style={'color': 'rgb(21, 99, 255)'},
                                                 className='count-card confirmed-case', selected_className='count-selected count-selected-case'),
                                         dcc.Tab(id='tab_death', label=f'{death_count}', value='Death', style={'color': 'rgb(237, 29, 48)'},
                                                 className='count-card confirmed-death', selected_className='count-selected count-selected-death')
                                     ]
                                     ),
                            className='p-0 tabs',
                        ),
                        dbc.Col(filters, md=7, sm=12,)
                    ],
                    className='h-50 tabs-filter-container'
                ),

                # Figures
                html.Div(
                    children=[
                               dbc.Row(
                    [
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id='map_plot',className='map', config={'displayModeBar': False}),lg=5, className='pr-0'),
                                dbc.Col(
                                    dbc.Row(
                                        [

                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            'Pays les plus touchés'),
                                                        dbc.CardBody(
                                                            dcc.Graph(id='top10', className='top-10-graph', config=config_dash))
                                                    ],
                                                    className='graph-card '),
                                                 lg=12
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            children='Evolution du nombre de cas', id='total_case_title'),
                                                        dbc.CardBody(
                                                            dcc.Graph(id='total_case_plot', className='total_case_plot',config=config_dash))
                                                    ],
                                                    className='graph-card new-case-card'),
                                                lg=12
                                            ),

                                        ],
                                        className='graphs'
                                    ),
                                className='graphs-container p-0',
                                sm=12, lg=7),
                                 dbc.Col(
                                    dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                            children='Nouveau cas', id='new_cases_title'),
                                                        dbc.CardBody(
                                                            dcc.Graph(id='new_cases', className='new', config=config_dash))
                                                    ], className='graph-card total-case-card'
                                                    ),
                                                
                                            )
                            ],
                            className='map-top10-row',
                            
                        )

                    ],
                    className='px-3 map-graph-row'
                ),
                    ],
                    className='data-display'
                )
         
            ],
            fluid=True,
        className='main-container'),
        # Footer
        html.Footer(
            children=[html.P(children='©️2020 Agensit')], className='footer')
    ]
)

@app.callback(
    [
        Output('my_title', 'children'),
        Output('my_date', 'children'),
        Output('tab_conf', 'label'),
        Output('tab_death', 'label'),
        Output('map_plot', 'figure'),
        Output('total_case_plot', 'figure'),
        Output('new_cases', 'figure'),
        Output('top10', 'figure'),
        Output('total_case_title', 'children'),
        Output('new_cases_title', 'children'),
    ],
    [
        Input('date_slider', 'value'),
        Input('tabs', 'value'),
        Input('country_dropdown', 'value')
    ]
)

def global_update(slider_date, tabs_type, country_dropdown):
# 0. Design
    # color and french legend
    if tabs_type == 'Death':
        marker_color = 'rgb(237, 29, 48)'
        type_value = 'morts'
    else:
        marker_color = 'rgb(21, 99, 255)'
        type_value = 'cas'
    colorized_elm = html.Span(children='COVID-19', style={'color': marker_color})

# 1. Preparation
    # global or detailed analysis ?
    if country_dropdown:
        df1 = df[df['State'].isin(country_dropdown)].reset_index(drop=True)
    else:
        df1 = df.copy()
    # filtred by date
    filtred_df = df1[df1['Date'] == df1['Date'][slider_date]].reset_index(drop=True)
    slice_df = df1[df1['Date'] <= df1['Date'][slider_date]].reset_index(drop=True)
    if country_dropdown:
        country_order = slice_df.groupby('State').sum().sort_values(by=tabs_type).index
    # create 'new_cases' and 'new_deaths'
    diff = slice_df.copy()
    diff['new_cases'] = diff['Confirmed'] - diff['Confirmed'].shift(1)
    diff['new_deaths'] = diff['Death'] - diff['Death'].shift(1)
    diff.dropna(inplace=True)
    # total count
    confirmed_count = filtred_df['Confirmed'].sum()
    death_count = filtred_df['Death'].sum()
    
# 2. MAP
    if country_dropdown:
        df_map = filtred_df[filtred_df['Death']>0].set_index('State')
        df_map = filtred_df.set_index('State')
        # zoom adjustment
        # mean_lat = df_map['Lat'].mean()
        # mean_lon = df_map['Long'].mean()
        # zoom_size= binning(df_map, country_dropdown)[0] - 1
        # plot
        map_plot = go.Figure([go.Scattermapbox(
            lat=[df_map.loc[c,'Lat']],
            lon=[df_map.loc[c,'Long']],
            customdata=[c],
            text=[millify(df_map.loc[c,tabs_type])],
            marker=dict(size=[df_map.loc[c,f'marker_{tabs_type}'] * 2], sizemin=3, sizeref=8),
            hovertemplate='<b>%{customdata}</b><br>' + '%{text}' + f' {type_value}' '<extra></extra>')
            for c in country_order])
        # map_plot.update_layout(mapbox={'zoom': zoom_size, 'center':dict(lat=mean_lat, lon=mean_lon)})
    else:
        if tabs_type == 'Death':
            df_map = filtred_df[filtred_df['Death']>0] 
        else:
            df_map = filtred_df.copy()
        # plot
        map_plot = go.Figure(go.Scattermapbox(
            lat=df_map['Lat'],
            lon=df_map['Long'],
            customdata=df_map['State'], 
            text=df_map[tabs_type].map(lambda x: millify(x)),
            marker_color=marker_color,
            marker=dict(size=df_map[f'marker_{tabs_type}'],sizemin=2, sizeref=8),
            hovertemplate='<b>%{customdata}</b><br>' + '%{text}' + f' {type_value}' '<extra></extra>'))
        # map_plot.update_layout(mapbox={'zoom': 0.4}) 

    map_plot.update_layout(hoverlabel=dict(bgcolor="white", font_size=12), margin=margin,
                           mapbox={'zoom':0.4, 'accesstoken': mapbox_access_token}, showlegend=False)
# 3. Top 10
    top10 = filtred_df.groupby(['State', 'Date']).sum().reset_index()
    top10 = top10.nlargest(10, tabs_type)
    top10.sort_values(tabs_type, inplace=True)
    if country_dropdown:
        top10.set_index('State',inplace=True)
        # plot
        top10_plot = go.Figure([go.Bar(
            x=[top10.loc[c,tabs_type]],
            y=[c], 
            text=[top10.loc[c, tabs_type]],
            textposition='outside',
            hovertemplate='%{text:.2s}' + f' {type_value}'+'<extra></extra>',
            orientation='h')
            for c in country_order])
    else:
        # plot
        top10_plot = go.Figure(go.Bar(
            x=top10[tabs_type],
            y=top10['State'],
            hovertemplate='%{text:.2s}' + 
            f' {type_value}'+'<extra></extra>',
            text=top10[tabs_type],
            marker_color=marker_color,
            textposition='outside',
            orientation='h'))
    top10_plot.update_layout(hovermode="y unified", showlegend=False, margin=margin)
    top10_plot.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    top10_plot.update_xaxes(showgrid=False, showticklabels=False)

 # 4. Cases over time
    if country_dropdown:
        global_increase = slice_df.groupby(['Date', 'State']).sum().reset_index(level='Date')
        total_case = go.Figure([go.Scatter(
            x=global_increase.loc[[c],'Date'].map(lambda x: pretty_date(x,'%d %B %Y')),
            y=global_increase.loc[[c], tabs_type],
            name=c)
            for c in country_order])
    else:
        global_increase = slice_df.groupby('Date').sum().reset_index()
        total_case = go.Figure(go.Scatter(
            x=global_increase['Date'].map(lambda x: pretty_date(x,'%d %B %Y')),
            y=global_increase[tabs_type],
            marker_color=marker_color))
    total_case.update_yaxes(showline=True)
    total_case.update_xaxes(showline=True, showgrid=False, showticklabels=False)
    total_case.update_layout(hovermode="x unified", margin=margin, showlegend=False)

# 5. New Cases Over time
    new_type = 'new_cases' if tabs_type == 'Confirmed' else 'new_deaths'
    if country_dropdown:
        global_diff = diff.groupby(['Date', 'State']).sum().reset_index(level='Date')
        # delete negative value
        global_diff.loc[global_diff['new_cases'] < 0,'new_cases'] = 0
        global_diff.loc[global_diff['new_deaths'] < 0,'new_deaths'] = 0
        # plot
        new_cases_plot = go.Figure([go.Bar(
            x=global_diff.loc[[c],'Date'].map(lambda x: pretty_date(x,'%d %B %Y')),
            y=global_diff.loc[[c], new_type],
            name=c)
            for c in country_order])
    else:
        global_diff = diff.groupby('Date').sum().reset_index()
        global_diff .loc[global_diff['new_cases'] < 0,'new_cases'] = 0
        global_diff.loc[global_diff['new_deaths'] < 0,'new_deaths'] = 0
        # plot    
        new_cases_plot = go.Figure(go.Bar(
            x=global_diff['Date'].map(lambda x: pretty_date(x,'%d %B %Y')),
            marker_color=marker_color,
            y=global_diff[new_type]))
    new_cases_plot.update_yaxes(showline=True)
    new_cases_plot.update_xaxes(showticklabels=False)
    new_cases_plot.update_layout(hovermode="x unified", showlegend=False,barmode='stack',margin=margin)

# 6. Output
    output_tuple = (
        ['Evolution du ', colorized_elm,' à travers le monde'],
        pretty_date(df['Date'][slider_date], '%d %B %Y'),
        f'{millify(confirmed_count)}',
        f'{millify(death_count)}',
        map_plot,
        total_case,
        new_cases_plot,
        top10_plot,
        f'Evolution du nombre total de {type_value}',
        f'Nombre de nouveau {type_value} par jour',
    )
    return output_tuple

if __name__ == "__main__":
    app.run_server(debug=True)