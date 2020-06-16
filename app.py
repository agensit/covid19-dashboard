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

df = pd.read_csv('database.csv')
df.drop(columns=['Country/Region','Recovered'], inplace=True)


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
        id='dropdown',
        options=[
            {'label': 'Nombre de mort', 'value': 'Death'},
            {'label': 'Nombre de cas', 'value': 'Confirmed'}],
        value='Confirmed'),
    ], style={'width':'100%', 'height':'100%'})
], className='h-75 p-4 mt-4')




# app = app.server
# Headers
app.layout = dbc.Container(
    [
    
        dbc.Card(dbc.Row([
            dbc.Col(html.H1(id='my_title', children='Evolution du COVID-19 à travers le monde'),md=8), 
            dbc.Col(html.H2(id='my_date'), md=4, className='text-right'),
        ],className='d-flex justify-content-between h-75'), className=' p-3'),
        
        dbc.Row([
            dbc.Col(dbc.Card(html.H2(id='confirmed_count', style={'width':'200px', 'height':'140px'})), sm= 12,md= 3, className="p-4"),
            dbc.Col(dbc.Card(html.H2(id='death_count', style={'width':'200px', 'height':'140px'})), sm= 12,md= 3, className="p-4"),
#Filters            
            dbc.Col(filters, md=6,className="mx-auto")
        ], className='h-50'),
      
        html.Hr(),
    
        # Filters
     
# Figures
    #Map & total_case_plot & new_cases
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(id ='map_plot', style={'padding':'3.9rem 3rem'})), sm=12, md=6), 
            dbc.Col([
                dbc.Card(dcc.Graph(id='total_case_plot',style={'height':'50%'})), 
                dbc.Card(dcc.Graph(id='new_cases'), className='mt-4')
            ],sm=12, md=6),
                ], className='m-4'), 
    #TOP 10 + Dash Table
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(id='top10')),sm=12, md=12, className="mt-4"),
        ]),
    

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
  

   

], fluid=True)

@app.callback(
    [Output('my_date', 'children'), 
    Output('confirmed_count', 'children'), 
    Output('death_count', 'children'),
    Output('map_plot', 'figure'),
    Output('total_case_plot','figure'),
    Output('new_cases', 'figure'),
    Output('top10', 'figure')],
    # Output('detailed_graph','figure')
   

    [Input('date_slider', 'value'),
    Input('dropdown', 'value'),])
    # Input('detailed_dropdown','value')])
    
# detailed_dropdown
def global_update(slider_date, dropdown_type):

# 0. Preparation

    # filtre df
    filtred_df = df[df['Date'] == df['Date'][slider_date]]
    slice_df = df[df['Date'] <= df['Date'][slider_date]] 
    # total count 
    confirmed_count = filtred_df['Confirmed'].sum() 
    death_count= filtred_df['Death'].sum() 
    # create new columns
    diff = slice_df.copy()
    diff['new_cases'] = diff['Confirmed'] - diff['Confirmed'].shift(1)
    diff['new_deaths'] = diff['Death'] - diff['Death'].shift(1)
    diff.dropna(inplace=True)

# 1. MAP
    map_df = filtred_df.copy()
    map_df["color"] = map_df[dropdown_type].map(lambda x: np.log2(x+1e-6)) 
    map_tooltip = {key:False for key,_ in map_df.items()}
    map_tooltip[dropdown_type]=True
    px.set_mapbox_access_token(mapbox_access_token)
    map_plot = px.scatter_mapbox(map_df, lat='Lat', lon='Long', 
                            hover_name = 'Province/State', hover_data = map_tooltip,
                            zoom = 0.4, #mapbox_style='dark',
                            size = 'Confirmed', size_max = 20,
                            #color = 'color', color_continuous_scale = ['Gold', 'DarkOrange', 'Crimson'],
                            width = 800, height = 800)

    map_plot.update_layout(hoverlabel=dict(bgcolor="white",font_size=12))
    map_plot.update(layout_coloraxis_showscale=True)

# 2. Cases over time
    global_increase = slice_df.groupby('Date').sum().reset_index()
    tooltip = {column:False for column in global_increase.columns}
    tooltip[dropdown_type] = True
    total_case = px.line(global_increase, x = 'Date', y = dropdown_type, 
        title = 'Nombre de cas en fonction du temps',hover_data = tooltip)
    total_case.update_yaxes(title=None)
    total_case.update_xaxes(title=None, showgrid=False)
    total_case.update_layout(hovermode="x unified")

# 3. New Cases Over time
    global_diff = diff.groupby('Date').sum().reset_index()
    global_diff = global_diff[global_diff['new_cases'] > 0]
    dropdown_new_type = 'new_cases' if dropdown_type == 'Confirmed' else 'new_deaths' # change the drop down type 
    tooltip = {column:False for column in global_diff}
    tooltip[dropdown_new_type] = True
    new_cases_plot = px.bar(global_diff, x='Date',y=dropdown_new_type, 
    	title='Nombre de nouveau cas en fonction du temps',hover_data=tooltip)
    new_cases_plot.update_yaxes(title=None)
    new_cases_plot.update_xaxes(title=None)
    new_cases_plot.update_layout(hovermode="x unified")

# 4. Top 10
    top10 = filtred_df.groupby(['Province/State', 'Date']).sum().reset_index()
    top10 = top10.nlargest(10,dropdown_type)
    top10.sort_values(dropdown_type, inplace=True)
    tooltip = {column:False for column in top10.columns}
    top10_plot = px.bar(top10, y='Province/State', x=dropdown_type, text=dropdown_type, orientation='h',
    	title='Les 10 pays avec le plus de cas',hover_name='Province/State', hover_data=tooltip)
    top10_plot.update_layout(hoverlabel=dict(bgcolor="white",font_size=12))
    top10_plot.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    top10_plot.update_yaxes(title=None )
    top10_plot.update_xaxes(title=None, showgrid=False, showticklabels=False)

# 5. Data per country
    # cases_per_country = diff[diff['Province/State'] == detailed_dropdown]
    # tooltip = {column:False for column in cases_per_country}
    # tooltip[dropdown_type] = True
    # cases_per_country_plot = px.bar(cases_per_country, x='Date',y=dropdown_type,
    #                     title= '{} - {}'.format(detailed_dropdown, dropdown_type), hover_data=tooltip)
    # cases_per_country_plot.update_yaxes(title=None)
    # cases_per_country_plot.update_xaxes(title=None)
    # cases_per_country_plot.update_layout(hovermode="x unified")

# Output
    output_tuple = (
        ' Date - {}'.format(df['Date'][slider_date]),
        'Nombre de cas {}'.format(confirmed_count),
        'Nombre de morts {}'.format(death_count),
        map_plot,
        total_case,
        new_cases_plot,
        top10_plot,
        # cases_per_country_plot
        )
    return output_tuple   

if __name__ == "__main__":
    app.run_server(debug=True)  

