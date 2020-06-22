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
from dash.dependencies import Output,Input



df = pd.read_csv('databasefr.csv')
df.drop(columns=['Country/Region','Recovered', 'Province/State'], inplace=True)

# create the counter
last_date = df['Date'].max()
confirmed_count = df[df['Date'] == last_date]['Confirmed'].sum()
death_count = df[df['Date'] == last_date]['Death'].sum() 


# discretization 
def discretize(serie, buckets):
    return pd.cut(serie.tolist(),buckets).codes

df['disc_Confirmed'] = discretize(df['Confirmed'].map(lambda x: x ** 0.4), 30)
df['disc_Death'] = discretize(df['Death'].map(lambda x: x ** 0.4), 30)

# create readable number
def millify(n):
    if n>999:
        if n > 1e6-1:
            return f'{round(n/1e6,1)}M'
        return f'{round(n/1e3,1)}K'
    return n



# set the date to french format
import locale
locale.setlocale(locale.LC_TIME, "fr_FR")
def pretty_date(str_date):
    date = parse(str_date)
    return date.strftime('%d %B %Y')




# mapbox token acess
with open('mapbox_token.txt') as f:
    lines=[x.rstrip() for x in f]
mapbox_access_token = lines[0]

### DASH APP


# added Bootstrap CSS.
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server



#filters component
filters = dbc.Card([
    html.Div([
        dcc.Slider(
        id = 'date_slider',
        min = 0,
        max = len(df['Date'].unique())-1,
        marks = {i:date for i,date in enumerate(df['Date'].unique()) if parse(date).day ==1},
        value = len(df['Date'].unique())-1),
        dcc.Dropdown(
        id = 'country_dropdown',
        options = [{'label': country, 'value': country} for country in df['State'].unique()],
        multi = True,
        value = None,
        className = 'country-dropdown-el')
], style={'width':'100%', 'height':'100%','padding':'.9rem'})
],className='filter-card')


# colorized title
titleSpanRed = html.Span(children='COVID-19', style={'color':'red'}) 

# death colorized

#confirmed_count colorized

# app = app.server

app.layout = html.Div([
    
    dbc.Container(
    [   
    # Headers
        dbc.Card(dbc.Row(
            [html.Div(children=[
                dbc.Col(html.H1(id='my_title', children=['Evolution du ', titleSpanRed, ' à travers le monde']),sm=12, md=8), 
                dbc.Col(html.H2(id='my_date',className='header-date'),sm=12, md=4),
            ],className='header')
            ]
            ,className='d-flex justify-content-between h-75 align-items-center'), className=' p-3 my-3'),
#Tabs
        html.Hr(),
        dbc.Row([
             dbc.Col( 
             dcc.Tabs(id="tabs", value='Confirmed', children=[
             dcc.Tab(id='tab_conf',label='{} Cas'.format(confirmed_count), value='Confirmed', style={'color':'red'},className='count-card confirmed-case', selected_className='count-selected'),
             dcc.Tab(id='tab_death',label='{} Morts'.format(death_count), value='Death', style={'color':'red'}, className='count-card confirmed-death', selected_className='count-selected'),
             ])),
            # dbc.Col(html.Div(html.H2(id='tab-conf', children=[confirmed_count, 'cas'], className='confirmed-count')), sm= 12,md= 3, className="p-3"),
            # dbc.Col(html.Div(html.H2(id='tab_death',children=[death_count, 'morts'], className='death-count')), sm= 12,md= 3, className="p-3"),
#Filters            
            dbc.Col(filters, md=6,className="mx-auto")
        ], className='h-50 p-1 mb-2'),
      
        html.Hr(),
    
        # Filters
     
# Figures
    #Map & total_case_plot & new_cases
        dbc.Row([
                dbc.Col([
                    dcc.Graph(id ='map_plot', className='map'),
                    dcc.Graph(id='top10', className='top-10-graph')
                ]), 
                dbc.Col([
                    dcc.Graph(id='total_case_plot'),
                    dcc.Graph(id='new_cases', className='new-cases'),
                ]),
            ],className='px-3 map-graph-row'), 
    #TOP 10 + Dash Table
   
    

# Detailed vizu
#     dbc.Row([
#         dbc.Col(html.Div([ html.H1(children='Exploration par pays')]), className='text-center mb-4'),
#     ]),
#     dbc.Row([
#         dbc.Col( dcc.Dropdown(
#         id = 'detailed_dropdown',
#         options = [{'label': country, 'value': country} for country in df['Province/State'].unique()],
#         multi = False,
#         value = 'Reunion'), md=2, className='mx-auto')
#     ]),
    
#    dcc.Graph(id='detailed_graph')
  



], fluid=True),

# Footer
    html.Footer(children=[html.P(children='©️2020 Agensit')],className='footer')
 ])

@app.callback(
    [Output('my_date', 'children'), 
    Output('tab_conf', 'label'),
    Output('tab_death', 'label'),
    Output('map_plot', 'figure'),
    Output('total_case_plot','figure'),
    Output('new_cases', 'figure'),
    Output('top10', 'figure'),
    ],

    [Input('date_slider', 'value'),
    Input('tabs', 'value'),
    Input('country_dropdown', 'value')
    ])
    

def global_update(slider_date, tabs_type, country_dropdown):

# 0. Preparation
    
    # filtre df
    if country_dropdown:
        df1 = pd.DataFrame([])
        for country in country_dropdown:
            df_country = df[df['State'] == country]
            df1 = pd.concat([df1,df_country])
        df1.reset_index(inplace=True)        
    else:
        df1 = df.copy()

    filtred_df = df1[df1['Date'] == df1['Date'][slider_date]]
    slice_df = df1[df1['Date'] <= df1['Date'][slider_date]] 
    # total count 
    confirmed_count = filtred_df['Confirmed'].sum() 
    death_count= filtred_df['Death'].sum() 
    # create new columns
    diff = slice_df.copy()
    diff['new_cases'] = diff['Confirmed'] - diff['Confirmed'].shift(1)
    diff['new_deaths'] = diff['Death'] - diff['Death'].shift(1)
    diff.dropna(inplace=True)

     # hoverinfo in french
    type_value = 'cas' if tabs_type == 'Confirmed' else 'morts'

# 1. MAP

    if country_dropdown:
        map_plot = go.Figure([go.Scattermapbox(
	        lat = filtred_df[filtred_df['State'] == country]['Lat'],
	        lon = filtred_df[filtred_df['State'] == country]['Long'],
	        customdata = filtred_df[filtred_df['State'] == country]['State'],
	        text = filtred_df[filtred_df['State'] == country][tabs_type].map(lambda x: millify(x)),
	        marker = go.scattermapbox.Marker(size = filtred_df[filtred_df['State'] == country][f'disc_{tabs_type}'] + 4, sizemin = 4),
	        hovertemplate = '<b>%{customdata}</b><br>' + '%{text}' + f' {type_value}' '<extra></extra>',
	        name = country) 
	        for country in country_dropdown])
    else:
	    map_plot = go.Figure(go.Scattermapbox(
	        lat = filtred_df['Lat'], 
	        lon = filtred_df['Long'],
	        customdata = filtred_df['State'],
	        text = filtred_df[tabs_type].map(lambda x: millify(x)),
	        marker = go.scattermapbox.Marker(size = filtred_df[f'disc_{tabs_type}']),
	        hovertemplate = '<b>%{customdata}</b><br>' + '%{text}' + f' {type_value}' '<extra></extra>'))


    map_plot.update_layout(hoverlabel=dict(bgcolor="white",font_size=12), margin=dict(l=0, r=0, t=0, b=0),
	                       mapbox = {'accesstoken': mapbox_access_token, 'zoom': 0.4}, showlegend = False)


    

 # 2. Cases over time 
    if country_dropdown:
        global_increase = slice_df.groupby(['Date','State']).sum().reset_index()
        total_case = go.Figure([go.Scatter(
        	x = global_increase[global_increase['State'] == country]['Date'],
            y = global_increase[global_increase['State'] == country][tabs_type],
            name = country)
            for country in country_dropdown])
    else:
        global_increase = slice_df.groupby('Date').sum().reset_index()

        total_case = go.Figure(go.Scatter(
        	x = global_increase['Date'], 
        	y = global_increase[tabs_type]))

    total_case.update_yaxes(title=None)
    total_case.update_xaxes(title=None)
    total_case.update_layout(hovermode="x unified", title=f'Evolution du nombre de {type_value}')

# 3. New Cases Over time
    new_type = 'new_cases' if tabs_type == 'Confirmed' else 'new_deaths' 

    if country_dropdown:
        global_diff = diff.groupby(['Date','State']).sum().reset_index()
        global_diff = global_diff[global_diff['new_cases'] > 0]
        global_diff = global_diff[global_diff['new_deaths'] > 0]

        new_cases_plot = go.Figure([go.Bar(
        	x = global_diff[global_diff['State'] == country]['Date'], 
            y = global_diff[global_diff['State'] == country][new_type], 
            name = country)
            for country in country_dropdown])
    else:
        global_diff = diff.groupby('Date').sum().reset_index()
        global_diff = global_diff[global_diff['new_cases'] > 0]
        global_diff = global_diff[global_diff['new_deaths'] > 0]

        new_cases_plot = go.Figure(go.Bar(
        	x = global_diff['Date'], 
        	y = global_diff[new_type]))

    new_cases_plot.update_layout(barmode='stack')
    new_cases_plot.update_yaxes(title=None)
    new_cases_plot.update_xaxes(title=None)
    new_cases_plot.update_layout(hovermode="x unified", title=f'Nouveau {type_value}', showlegend=False)

# 4. Top 10
    top10 = filtred_df.groupby(['State', 'Date']).sum().reset_index()
    top10 = top10.nlargest(10,tabs_type)
    if country_dropdown:
        top10_plot = go.Figure([go.Bar(
        	x = top10[top10['State'] == country][tabs_type], 
            y = top10[top10['State'] == country]['State'], name=country, 
            text = top10[top10['State'] == country][tabs_type],
            textposition = 'outside',
            hovertemplate = '<b>%{y}</b><br>' + '%{text:.2s}'+f' {type_value}'+'<extra></extra>', 
            orientation = 'h')
            for country in country_dropdown])
    else:
        top10.sort_values(tabs_type, inplace=True)
        top10_plot = go.Figure(go.Bar(
        	x = top10[tabs_type], 
        	y = top10['State'], 
            # hoverinfo = 'text+y',
            hovertemplate = '<b>%{y}</b><br>' + '%{text:.2s}'+f' {type_value}'+'<extra></extra>', 
            text = top10[tabs_type],
            textposition = 'outside',
            orientation = 'h'))

    top10_plot.update_layout(hoverlabel=dict(bgcolor="white",font_size=12), title='Pays les plus touchés', showlegend=False)
    top10_plot.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    top10_plot.update_yaxes(title=None )
    top10_plot.update_xaxes(title=None, showgrid=False, showticklabels=False)


# Output
    output_tuple = (
        pretty_date(df['Date'][slider_date]),
        f'{millify(confirmed_count)} cas',
        f'{millify(death_count)} morts',
        map_plot,
        total_case,
        new_cases_plot,
        top10_plot,
        )
    return output_tuple 

if __name__ == "__main__":
    app.run_server(debug=True)  

