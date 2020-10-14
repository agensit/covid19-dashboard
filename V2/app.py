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


'''
   ------------------------------------------------------------------------------------------- 
                                            CONFIG
   ------------------------------------------------------------------------------------------- 
'''
# ## TESTE RESOLUTION
# import gtk
# width = gtk.gdk.screen_width()
# height = gtk.gdk.screen_height()

## PLOTLY
import plotly.io as pio
pio.templates.default = "plotly_white"
# mapbox token acess
with open('mapbox_token.txt') as f:
    lines = [x.rstrip() for x in f]
mapbox_access_token = lines[0]

## DASH
config_dash = {'displayModeBar': False, 'showAxisDragHandles':False}  
margin = dict(l=10, r=10, t=10, b=10)
# External CSS + Dash Bootstrap components
external_stylesheets=[dbc.themes.BOOTSTRAP, "assets/main.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

## USE THE FRENCH DATE - not working with heroku
import locale
locale.setlocale(locale.LC_TIME, "fr_FR")

## READ BIG NUMBER
def millify(n):
    if n > 999:
        if n > 1e6-1:
            return f'{round(n/1e6,1)}M'
        return f'{round(n/1e3,1)}K'
    return n

def format_date(str_date, date_format):
    date = parse(str_date)
    return date.strftime(date_format) 

## Colors 
red = '#ed1d30'
blue = '#2e72ff'

'''
   ------------------------------------------------------------------------------------------- 
                                            LOAD THE DATA
   ------------------------------------------------------------------------------------------- 

'''
covid19 = pd.read_csv('collect_data/df_covid19.csv')

countries =  covid19['State'].unique()
dates = covid19['Date'].unique()
last_date = dates.max()
cases_counter = covid19.loc[covid19['Date'] == last_date, 'Confirmed'].sum()
death_counter = covid19.loc[covid19['Date'] == last_date, 'Death'].sum()


'''------------------------------------------------------------------------------------------- 
                                        DASH COMPONENTS
   ------------------------------------------------------------------------------------------- 
'''
## FILTERS
filters = dbc.Card([
    dcc.RangeSlider(
        id='date_slider',
        count=1,
        min=0,
        max= len(dates)-1,
        marks={i: format_date(date,'%B').title() for i, date in enumerate(dates) if parse(date).day == 1},
        value=[0, len(dates) - 1],
        className='mt-3 mb-3'), 
    dcc.Dropdown(
        id='country_dropdown',
        options=[{'label': country, 'value': country} for country in countries],
        multi=True,
        value=None,
        placeholder='Choisir un pays',
        className='mb-3 ml-2 mr-2')
])

## TABS
tabs = dcc.Tabs(
    id="tabs", 
    value='Confirmed', 
    children = [
        dcc.Tab(
            id='tab_conf', 
            label=f'{cases_counter}', 
            value='Confirmed', 
            style={'color': blue},
            className='count-card confirmed-case', 
            selected_className='count-selected count-selected-case'),
        dcc.Tab(
            id='tab_death', 
            label=f'{death_counter}', 
            value='Death', 
            style={'color': red},
            className='count-card confirmed-death', 
            selected_className='count-selected count-selected-death')
    ]
),

## FIGURES
# Map
map_container = dbc.Card(
    children  = [
        dcc.Graph(
            id='map_plot', 
            config={**config_dash, **{'scrollZoom':False}})
    ],
    className='mr-0 mt-0'
)
# total cases
total_case_container = dbc.Card(
    children = [
        html.H5(id = 'total_case_title', className = "mt-2 ml-2 mb-0"),
        dcc.Graph(id = 'total_case_plot', config = config_dash),
    ],
    # style={'height':'50vh'}
)
# top 10
top10_container = dbc.Card(
    children = [
        html.H5('Pays les plus touchés', className="mt-2 ml-2 mb-0"),
        dcc.Graph(id='top10', config=config_dash)
    ]
)
# daily cases
daily_cases_container = dbc.Card(
    children = [
        html.H5(id='daily_cases_title',className="mt-2 ml-2 mb-0"),
        dcc.Graph(id='new_cases', config=config_dash)
    ]
)

'''------------------------------------------------------------------------------------------- 
                                            LAYOUT
   ------------------------------------------------------------------------------------------- 
'''
app.layout = html.Div(
    [
        dbc.Container(
            children = [
                # Headers
                dbc.Row(
                    children = [
                        dbc.Col(html.H2(id='my_title'), width =9), 
                        dbc.Col(html.H4(id='my_date'), width=3)
                    ],
                    align="end",
                    justify='between',
                    className='mt-2 mb-0'),

                # Tabs & Filters
                dbc.Row(
                    children =[
                        dbc.Col(tabs, className='mr-3'), 
                        dbc.Col(filters),
                    ],
                no_gutters=True,
                align="center"),

                # Map & total_cases
                dbc.Row(
                    children = [
                        dbc.Col(map_container, className='mr-3'),
                        dbc.Col(total_case_container)
                    ],
                    no_gutters=True,
                    className='mb-3 mt-3'),

                # top10 + daily_cases
                dbc.Row(
                    children= [
                        dbc.Col(top10_container, className='mr-3'),  
                        dbc.Col(daily_cases_container)
                    ],
                    no_gutters=True,
                    className='mb-2'),

                # Footer
                dbc.Row(html.P(children='©️2020 Agensit'), justify='center')
            ],
            fluid=True,
            # className='main-container'   
        ),
    ]
)

'''------------------------------------------------------------------------------------------- 
                                            INTERACT
   ------------------------------------------------------------------------------------------- 
'''

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
        Output('daily_cases_title', 'children'),
    ],
    [
        Input('date_slider', 'value'),
        Input('tabs', 'value'),
        Input('country_dropdown', 'value')
    ]
)

def global_update(slider_date, tabs_type, country_dropdown):

# 0. DESIGN
# --------------------------------------------------------
    # date panel (top - right)
    starting_date = covid19['Date'][slider_date[0]]
    ending_date = covid19['Date'][slider_date[1]]
    min_date_panel = format_date(starting_date , '%d %B')
    max_date_panel = format_date(ending_date, '%d %B %Y')

    # color and french legend
    if tabs_type == 'Death':
        marker_color = red
        type_value = 'morts'
    else:
        marker_color = blue
        type_value = 'cas'
    colorized_elm = html.Span(children='COVID-19', style={'color': marker_color})

# 1. PREPARATION
# --------------------------------------------------------
    # filtre by country
    # -----------------
    if country_dropdown:
        df = covid19[covid19['State'].isin(country_dropdown)].reset_index(drop=True)
    else:
        df = covid19.copy()
    
    # filtre by date
    # --------------
    min_range_slider = df.loc[df['Date'] == starting_date].reset_index(drop=True)
    max_range_slider = df[df['Date'] == ending_date].reset_index(drop=True)

    # table with last slider date
    filtred_df = max_range_slider.copy()
    numeric_column = ['Confirmed', 'Death']
    filtred_df[numeric_column] = max_range_slider[numeric_column] - min_range_slider[numeric_column] 

    # table with date between slider date values
    slice_df = df[df['Date'] <= df['Date'][slider_date[1]]]
    slice_df = slice_df[slice_df['Date'] >= slice_df['Date'][slider_date[0]]].reset_index(drop=True)

    # key values
    # ----------
    cases_counter = filtred_df['Confirmed'].sum()
    death_counter = filtred_df['Death'].sum()

    if country_dropdown:
        country_order = slice_df.groupby('State').sum().sort_values(by=tabs_type).index
    
# 2. MAP
# --------------------------------------------------------
    df_map = filtred_df.copy()
    # filtering by country
    if country_dropdown: 
        df_map = filtred_df[filtred_df['Death']>0].set_index('State')
        df_map = filtred_df.set_index('State')
        map_plot = go.Figure()
        for c in country_order:
            map_plot.add_traces(
                go.Scattermapbox(
                    lat = df_map.loc[c,['Lat']],
                    lon = df_map.loc[c,['Long']],
                    text = [c],
                    customdata = np.dstack((df_map.loc[c,'Confirmed'],df_map.loc[c,'Death']))[0],
                    marker = dict(size = 10),
                    hovertemplate = '<b>%{text}</b><br><br>' + '%{customdata[0]:.3s} cas<br>' + '%{customdata[1]:.3s} morts<extra></extra>'
                )
            )
    # global cases
    else:
        # hide no death in map when you filter by death 
        if tabs_type == 'Death':
            df_map = filtred_df[filtred_df['Death']>0] 

        # set the marker size
        bubble_size = df_map.set_index('State')[tabs_type]
        bubble_size[bubble_size < 0] = 0
        # plot
        map_plot = go.Figure(
            go.Scattermapbox(
                lat = df_map['Lat'],
                lon = df_map['Long'],
                customdata = np.dstack((df_map['Confirmed'],df_map['Death']))[0], 
                text = df_map['State'],
                marker_color = marker_color,
                marker = dict(
                    size =  bubble_size,
                    sizemode = 'area',
                    sizemin = 1, 
                    sizeref = 2. * max(bubble_size) / (40.**2),
                ),
                hovertemplate='<b>%{text}</b><br><br>' + '%{customdata[0]:.3s} cas<br>' + '%{customdata[1]:.3s} morts<extra></extra>'
            )
        )

    # figure design
    map_plot.update_layout(
        hoverlabel = dict(
            bgcolor = "white", 
            font_size = 12), 
        margin = margin,
        mapbox = dict(
            zoom = 0, 
            style = 'mapbox://styles/axelitorosalito/ckdyhbsb93rp719mwude0ze6j',
            center = go.layout.mapbox.Center(
                lat = 30,
                lon = 0
            ),
            accesstoken = mapbox_access_token
        ), 
        showlegend = False
    )

# 3. Top 10
# --------------------------------------------------------
    top10 = filtred_df.groupby(['State', 'Date']).sum().reset_index()
    if country_dropdown:
        top10 = top10.nlargest(len(country_dropdown), tabs_type)
        top10.sort_values(tabs_type, inplace=True)
        top10.set_index('State',inplace=True)
        # plot
        top10_plot = go.Figure()
        for c in country_order:
            top10_plot.add_traces(go.Bar(
                x=[top10.loc[c,tabs_type]],
                y=[c], 
                customdata=np.dstack((top10.loc[c,'Confirmed'],top10.loc[c,'Death']))[0],
                hovertemplate='%{customdata[0]:.3s} cas<br>' +
                '%{customdata[1]:.3s} morts <extra></extra>'))
    else:
        top10 = top10.nlargest(10, tabs_type)
        top10.sort_values(tabs_type, inplace=True)
        # plot
        top10_plot = go.Figure(go.Bar(
            x=top10[tabs_type],
            y=top10['State'],
            customdata=np.dstack((top10['Confirmed'],top10['Death']))[0],
            hovertemplate='%{customdata[0]:.3s} cas<br>' +
            '%{customdata[1]:.3s} morts <extra></extra>',
            marker_color=marker_color))
    top10_plot.update_layout(hovermode="y unified", showlegend=False, margin= margin)
    top10_plot.update_traces(texttemplate='%{customdata[0]:.3s}', textposition='outside', orientation='h')
    top10_plot.update_xaxes(showgrid=False, showticklabels=False)
    top10_plot.update_yaxes(linewidth=0.5, linecolor='black')

# 4. Cases over time
# --------------------------------------------------------
    # filtering by country
    if country_dropdown:
        global_increase = slice_df.groupby(['Date', 'State']).sum().reset_index(level='Date')
        total_case = go.Figure()
        for c in country_order:
            total_case.add_traces(
                go.Scatter(
                    x = global_increase.loc[[c],'Date'].map(lambda x: format_date(x,'%d %b %y')),
                    y = global_increase.loc[[c], tabs_type],
                    name = c
                )
            )

    # global cases
    else:
        global_increase = slice_df.groupby('Date').sum().reset_index()
        total_case = go.Figure(
            go.Scatter(
                x = global_increase['Date'].map(lambda x: format_date(x,'%d %b %y')),
                y = global_increase[tabs_type],
                customdata = global_increase[tabs_type],
                hovertemplate = '%{customdata:.3s}<extra></extra>',
                marker_color = marker_color
            )
        )

    # figure design
    total_case.update_yaxes(showline=True, nticks=5)
    total_case.update_xaxes(showline=False, nticks=5, showgrid=True)
    total_case.update_layout(hovermode="x unified", margin=margin, showlegend=False)

# 5. Daily Cases
# --------------------------------------------------------
    # create 'new_cases' and 'new_deaths'
    daily_cases = slice_df.copy()
    columns = ['Confirmed', 'Death']
    def comput_diff(df, columns=columns):
        return df[columns] - df[columns].shift(1)
    daily_cases[['new_cases', 'new_deaths']] = daily_cases.groupby('State').apply(comput_diff)
    daily_cases.fillna(0, inplace=True)
    new_type = 'new_cases' if tabs_type == 'Confirmed' else 'new_deaths'
    daily_cases['Date'] = pd.to_datetime(daily_cases['Date'])
    daily_cases.set_index('Date', inplace=True)

    # filtering by country
    if country_dropdown:
        new_cases_plot = go.Figure()
        for c in country_order:
            country_daily_cases = daily_cases[daily_cases['State'] == c]
            country_daily_cases = country_daily_cases.resample('7D').sum() # resample in a weekly base
            evolution = country_daily_cases[new_type].map(lambda x: x / (country_daily_cases[new_type].sum() + 1e-5)) # 1e-5 --> prevent deviding by 0

            new_cases_plot.add_traces(
                go.Scatter(
                    x = evolution.index,
                    y = evolution,
                    name = c,
                    customdata = country_daily_cases[new_type],
                    hoverinfo='skip',
                    fill = 'tozeroy',
                    line_shape = 'spline'
                )
            )

    # global cases
    else:
        daily_cases = daily_cases.groupby('Date').sum()
        daily_cases = daily_cases.resample('7D').sum()

        new_cases_plot = go.Figure(
            go.Scatter(
                x = daily_cases.index,
                y = daily_cases[new_type],
                hoverinfo='skip',
                marker_color = marker_color,
                fill = 'tozeroy',
                line_shape = 'spline',
            )
        )
    # figure design
    new_cases_plot.update_yaxes(showgrid=False, nticks=5, showticklabels=False, )
    new_cases_plot.update_xaxes(showline=True, nticks=5, showgrid=True, zeroline=False)
    new_cases_plot.update_layout(hovermode="x unified", showlegend=False, margin=margin)

# 6. Output
# --------------------------------------------------------
    output_tuple = (
        ['Evolution du ', colorized_elm,' à travers le monde'],
        '{} au {}'.format(min_date_panel, max_date_panel),
        f'{millify(cases_counter)}',
        f'{millify(death_counter)}',
        map_plot,
        total_case,
        new_cases_plot,
        top10_plot,
        f'Evolution du nombre total de {type_value}',
        f'Taux de nouveau {type_value}'
    )
    return output_tuple

if __name__ == "__main__":
    app.run_server(debug=True)