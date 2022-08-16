# data manipulation
from dash.html.Title import Title
import numpy as np
import pandas as pd
import math

# plotly 
import plotly.express as px
import plotly.graph_objects as go

# dashboards
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from datetime import date
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# app = JupyterDash(__name__, external_stylesheets=[dbc.themes.UNITED])
# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI])
server = app.server  # expose server variable for Procfile

# read the data
df = pd.read_csv('df_sub_.csv')
data_ = df[df.room_type!='Entire home/apt'].copy()
stat = df.describe()
# table=pd.read_csv('table.csv')

df_a = [
    'host_total_listings_count','host_is_superhost','bedrooms', 
    'beds','number_of_reviews', 'minimum_nights',
    'median_income','availability_365','review_scores_rating',
    'num_amenities','log_median_income'
]
df_b = [
    'host_total_listings_count','host_is_superhost','bedrooms', 
    'beds','number_of_reviews', 'minimum_nights',
    'median_income','availability_365','review_scores_rating',
    'price','log_price' ,'num_amenities','log_median_income'
]
available_indicators = ['price','log_price']
columns = ['bedrooms','beds','number_of_reviews','minimum_nights','availability_365','review_scores_rating',
        'num_amenities','log_median_income']

radio_room_type = dbc.RadioItems(
        id='room_type', 
        className='radio',
        options=[dict(label='Private Room', value='Private room'), dict(label='Shared Room', value='Shared room'), dict(label='Entire Home', value='Entire home/apt'), dict(label='All Types', value='All types')],
        value='All types', 
        inline=True,
    )

app_text='The app explores the relationship between the listing PRICE for each home in airbnb and different variables, and develops a smart system to automatically give suggested price and its interval.'

intro_text='Select the type of each home from radioitem, click on the button to visualize output regions geographically on the map, computing may take seconds to finish. Click on the node on the map to check the Host ID, listed price, the median income in corresponding zip code, the number of amenities, and the review scores rating for each home. By clicking the button from radioitem, it shows the percentage number of homes for each room type out of the total of listings, the average price for the selected type, and the median price. The distribution graph shows the lognormal distribution of price for selected type. The X-axis denotes values of nature log of price for each type of room. The Y-axis denotes the frequency of appearance of nature log of prices.'

box_text=' By clicking the button from radioitem, it shows the percentage number of homes for each room type out of the total of listings, the average price for the selected type, and the median price. The distribution graph shows the lognormal distribution of price for selected type. The X-axis denotes values of nature log of price for each type of room. The Y-axis denotes the frequency of appearance of nature log of prices. '

# amenities word
words = []
with open('amenities.txt', 'r') as f:
    for line in f.readlines():
        words.append(line.replace('"','').replace(', \n','').replace(',\n',''))
f.close()

# lr-regression
def lr_default(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13):
    """
    v1: room_type[t.Private room]
    v2: room_type[T.Shared room]  
    v3: review_scores_rating 
    v4: number_of_reviews
    v5: review_scores_rating:number_of_reviews
    v6: beds
    v7: bedrooms
    v8: log_median_income
    v9: host_total_listings_count
    v10: host_is_superhost
    v11: minimum_nights
    v12: num_amenities
    v13: availability_365
    """
    coef = np.array([-0.6268,-1.2267,0.1113,-0.0160,0.0032,0.0116,0.3577,0.3823,4.813e-05,0.0195,-0.0010,0.0050,0.0005])
    value = np.array([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13])
    intercept = -0.4853
    result = np.sum(coef*value)+intercept
    return result


# confidence interval
def ci_cal(pred,rmse=0.5210,n=19270):
    '''
    set t-statistic to be 1.96 since the total observation > 100
    '''
    upper = pred + 1.96 * rmse / math.sqrt(n)
    lower = pred - 1.96 * rmse / math.sqrt(n)
    return [lower,upper]

# data inputs
controls = [
    dbc.Row([
        dbc.Label("Room_type", style={"color": "rgb(72,72,72)"}),
        dbc.RadioItems(
            inline=True,
            id='room_type-input',
            options=[
                {"label": "Private room", "value": 'Private room'},
                {"label": "Shared room", "value": 'Shared room'},
                {"label": "Entire home/apt", "value": 'Entire home/apt'},
                {"label": "Hotel room", "value": 'Hotel room', "disabled": True}
            ],
            # options=[{"label": v, "value": v} for v in data.room_type.unique()],
            value="Private room",
            labelStyle={"display": "inline-block"},
        )
    ]),
    dbc.Row([
        dbc.Label('Review_scores_rating', style={"color": "rgb(72,72,72)"}),
        dcc.Slider(
            id='rs_rating_input',
            min=0,
            max=5,
            value=4.7,
            step=0.01,
            marks={i: str(i) for i in range(0, 6, 1)},
        )
    ]),
    dbc.Row([
        dbc.Label('Number_of_reviews', style={"color": "rgb(72,72,72)"}),
        dcc.Slider(
            id="number_of_reviews",
            min=0,
            max=df.number_of_reviews.max(),
            value=200,
            step=1,
            marks={i: str(i) for i in range(0, df.number_of_reviews.max()+1, 100)},
        ),
    ]),
    dbc.Row([
        dbc.Label('Beds', style={"color": "rgb(72,72,72)"}),
        dcc.Slider(
            id="beds",
            min=0,
            max=df.beds.max(),
            value=5,
            step=1,
            marks={i: str(i) for i in range(0, int(df.beds.max()+1), 5)},
        ),
    ]),
    dbc.Row([
        dbc.Label('Bedrooms', style={"color": "rgb(72,72,72)"}),
        dcc.Slider(
            id="bedrooms",
            min=0,
            max=df.bedrooms.max(),
            value=4,
            step=1,
            marks={i: str(i) for i in range(0, int(df.bedrooms.max()+1), 5)},
        ),
    ]),
    dbc.Row([
        dbc.Label('Median_income', style={"color": "rgb(72,72,72)"}),
        dcc.Slider(
            id="median_income",
            min=0,
            max=df.median_income.max(),
            value=65000,
            step=100,
            marks={i: str(i) for i in range(0, int(df.median_income.max()+1), 30000)},
        ),
    ]),
    dbc.Row([
        dbc.Label('Host_total_listings_count', style={"color": "rgb(72,72,72)"}),
        dcc.Slider(
            id="h_t_l_count",
            min=0,
            max=df.host_total_listings_count.max(),
            value=50,
            step=5,
            marks={i: str(i) for i in range(0, int(df.host_total_listings_count.max()+1), 200)},
        )
    ]),
    dbc.Row([
        dbc.Label('Host_is_superhost', style={"color": "rgb(72,72,72)"}),
        dbc.RadioItems(
            inline=True,
            id="host_is_superhost",
            options=[{"label": v, "value": v} for v in df.host_is_superhost.unique()],
            value=df.host_is_superhost[0],
            labelStyle={"display": "inline-block"},
        )
        # id="control-item-host",
    ]),
    dbc.Row([
        dbc.Label('Minimum_nights', style={"color": "rgb(72,72,72)"}),
        dcc.Slider(
            id="minimum_nights",
            min=0,
            max=df.minimum_nights.max(),
            value=200,
            step=1,
            marks={i: str(i) for i in range(0, df.minimum_nights.max()+1, 100)},
        )
    ]),
    dbc.Row([
        dbc.Label('Availability_365', style={"color": "rgb(72,72,72)"}),
        dcc.Slider(
            id="availability_365",
            min=0,
            max=df.availability_365.max(),
            value=df.availability_365[0],
            step=1,
            marks={i: str(i) for i in range(0, df.availability_365.max()+1, 50)},
        )
    ]),
    # TO DO LIST: add amenities here
    dbc.Row([
        dbc.Label("Amenities", style={"color": "rgb(72,72,72)"}),
        dbc.Col([
            dbc.Checklist(
                id="select-1",
                options=[{"label": v, "value": v} for v in words[0:int(len(words)/2+1)]],
                value = ['Shampoo','Coffee maker','Free parking on premises','Gym']
            ),
        ],md=6),
        dbc.Col([
            dbc.Checklist(
                id="select-2",
                options=[{"label": v, "value": v} for v in words[int(len(words)/2+1):]],
                value = ['Backyard','Wifi','Oven','Heating']
            ),
        ],md=6),
    ])
    # dbc.Row()
]

app.layout = html.Div([

    html.Img(
            src=app.get_asset_url('./airbnb1.png'),
            style={
                "height": "50px",
                "width": "auto",
                "margin-bottom": "0px",
            }),

    html.H1('Airbnb Smart Pricing', style={'text-align': 'center', 'color': 'black', "font-weight": "bold"}),

    html.Br(),
    
    html.Div([
        html.H5('About this app',style={"font-weight": "bold"}),
        html.P(app_text),
        html.P(intro_text,),
    ], style={'display': 'inline-block', 'margin-left': '30px','margin-right': '30px'}),

    html.Div(children=[
        html.Div([
            html.Br(),
            html.Label("Choose the Room Type:"), 
            html.Br(),
            html.Br(),
            radio_room_type
        ], className='box', style={'padding-bottom':'15px'})
    ],style={'width': '48%', 'display': 'inline-block', 'flex': 1, "margin-left": "30px"}),

    html.Div(children=[
        dcc.Graph(id= 'map', 
                  figure={}, 
                  style={"margin": "4px", 'border-style':'outset', 'width':'49%'},
                 ),
 
        html.Div(className="row", children=[
            html.Br(),
            html.Div([
                html.Div([
                    html.H4('Count %', style={'font-weight':'normal'}),
                    html.H3(id='count')
                ],className='box_box', style={"border":"2px black solid", "margin-left": "25px", 
                                              "padding-top": "15px", "padding-bottom": "15px",
                                              "padding-left": "15px", "padding-right": "15px",
                                              "box-shadow": "5px 5px #F66F9F",
                                              "border-color": "#F66F9F",
                                              "margin": "7px",  'width':'33%'}),

                html.Div([
                    html.H4('Average Price', style={'font-weight':'normal'}),
                    html.H3(id='avg_price')
                ],className='box_box', style={"border":"2px black solid", "margin-left": "25px", 
                                              "padding-top": "15px", "padding-bottom": "15px",
                                              "padding-left": "15px", "padding-right": "15px",
                                              "box-shadow": "5px 5px #F66F9F",
                                              "border-color": "#F66F9F",
                                              "margin": "7px" , 'width':'33%'}),
            
                html.Div([
                    html.H4('Median Price', style={'font-weight':'normal'}),
                    html.H3(id='med_price')
                ],className='box_emissions', style={"border":"2px black solid", "margin-left": "25px", 
                                              "padding-top": "15px", "padding-bottom": "15px",
                                              "padding-left": "15px", "padding-right": "15px",
                                              "box-shadow": "5px 5px #F66F9F",
                                              "border-color": "#F66F9F",
                                              "margin": "7px", 'width':'33%'}),
            ],style={'display': 'flex', 'border-style':'outset'}),

            html.Br(),
            html.Br(),
            
            html.Div(children=[
                dcc.Graph(id='log_priceDist', figure={}),
            ], style={'border-style':'outset', 'display': 'inline-block', 'margin-top':'10px'})

        ],  style={'padding': 5, 'flex': 1, 'width': '49%', "margin-left": "5px"})

    ],className='box',style={'display': 'flex', 'flex-direction': 'row', 'margin': '20px',}),

    html.Br(),
    ##part 2
    
    html.Div([
        html.H2('Key Metrics about Data', style = {'text-align': 'center', 'font-weight': 'bold', 'font-size':'20px'}),
        dcc.Dropdown(id = 'describe_stats_drop_down',
                    options=[{'label': i, 'value': i} for i in ['price','number_of_reviews','review_scores_rating','host_total_listings_count',
                             'bedrooms','beds','minimum_nights','num_amenities','median_income']],
                     value = 'price'),
        dcc.Graph(id = 'describe_stats_table')
            ],style={'width': '27%', 'display': 'inline-block', 'margin-left': '25px', 
                          'padding':'15px', 'border-style':'outset'}),
    html.Div([
        html.H2('Correlation Analysis for Price', style = {'text-align': 'center', 'font-weight': 'bold', 'font-size':'20px'}),
        # top left drop menu and radio items together in one division
        dcc.Dropdown(id='measurement',
            options=[{'label': i, 'value': i} for i in available_indicators],
            value='log_price'),
        dcc.Graph(id='corr_plot'),
        dcc.Graph(id='radar_plot')
            ], style={'width': '27%', 'display': 'inline-block', 'margin-left': '2px', 
                  'padding':'15px', 'border-style':'outset'}),
    
    
    html.Div([
        html.H2('Regression Analysis for Price', style = {'text-align': 'center', 'font-weight': 'bold', 'font-size':'20px'}),
        dcc.Dropdown(id='x_dropdown',
                     options=[{'label': x_val, 'value': x_val}
                              for x_val in columns],
                    value = 'review_scores_rating'),
        dcc.Graph(id='scatter_output',
                 hoverData={'points': [{'customdata': 109}]})
            ], style={'width': '43%','display': 'inline-block', 'margin-left': '2px',
                      'padding':'15px', 'border-style':'outset'}),

    dbc.Container(
        [
            html.H2("Smart Pricing for Airbnb with Confidence Interval Estimation", 
                        className="text-center mt-4 mb-3",
                        style={"color": "rgb(255,90,95)", "text-decoration": "None",'fontWeight': 'bold'},
                    ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(
                                    [dbc.CardHeader("Model Input", style={"color": "rgb(72,72,72)"}), 
                                    dbc.CardBody(controls)]
                                    ), 
                                    md=4),
                    dbc.Col(
                        md=8,
                        children=[
                            dbc.Row('Automatic Report of Pricing Strategy and Solution', style={"color": "rgb(72,72,72)"}),
                            # html.Br(),
                            html.Hr(),
                            # dcc.Markdown(
                            #     id=test_output
                            # )
                            html.Div(id='test_output',className="card-text",style={'color':'rgb(118,118,118)'}),
                            html.Div(id='test_output-2',className="card-text",style={'color':'rgb(118,118,118)'}),
                            html.Br(),
                            dbc.Row('The Attributes and Distribution within houses in the above confidence interval', style={"color": "rgb(72,72,72)"}),
                            # html.Br(),
                            html.Hr(),
                            dcc.Graph(
                                    id="model-output",
                                    # figure=go.Figure(),
                                    style={"height": "calc(100vh - 150px)", "min-height": "500px"},
                            ),
                            dcc.Graph(
                                    id="model-output-2",
                                    # figure=go.Figure(),
                                    style={"height": "calc(60vh - 150px)", "min-height": "500px"},
                            )
                            # dcc.Col(
                            #     children = 
                            # )
                        ],
                    ),
                ],
            ),
        ],
        fluid=True
    ),
        ##summary
    html.Div(
        children=[
            html.H5('Sources', style={'text-align': 'center', 'color': 'black', "font-weight": "bold"}),
            html.Ul([
                html.Li(['US Census Buearue:  ', 
                        html.A('https://data.census.gov/cedsci/table?q=income&g=0100000US%248600000&tid=ACSST5Y2019.S1901&hidePreview=true',
                        href='https://data.census.gov/cedsci/table?q=income&g=0100000US%248600000&tid=ACSST5Y2019.S1901&hidePreview=true'),
                        ]),
                html.Li(['Inside Airbnb:  ', 
                        html.A('http://insideairbnb.com/get-the-data.html',
                        href='http://insideairbnb.com/get-the-data.html'),
                        ]),
            ])
        ], style={'border-style':'outset', "margin":"20px"} ),
    
    html.Div(
        children=[
            html.H5('Authors', style={'text-align': 'center', 'color': 'black', "font-weight": "bold"}),
            html.P('Zongqian Wu, Yuxuan Yuan, Luyue Zhang, Yijia Wang, Shuran Fu, Hongyi Shao', style={'text-align': 'center', 'color': 'black'})
        ], style={'border-style':'outset', "margin":"20px"} ),

])



@app.callback([Output(component_id='count', component_property='children'),
            Output(component_id='avg_price', component_property='children'),
            Output(component_id='med_price', component_property='children')],
              Input(component_id='room_type', component_property='value'))

def update_box(roomtype_choice):
    if roomtype_choice=='All types':
        df_sub=df
    else:
        df_sub=df[df['room_type']==roomtype_choice]
    
    count_str = str(np.round(df_sub['room_type'].count()/df['room_type'].count(),4)*100)+str('%')
    avg_price_str = str(np.round(df_sub["price"].mean(),2))
    med_price_str = str(np.round(df_sub["price"].median(),2))

    return count_str, avg_price_str, med_price_str

@app.callback([Output(component_id='map', component_property='figure'),
              Output(component_id='log_priceDist', component_property='figure')],
              Input(component_id='room_type', component_property='value'))

def update_graph(roomtype_choice):
    if roomtype_choice=='All types':
        df_sub=df
    else:
        df_sub=df[df['room_type']==roomtype_choice]

    fig1 = px.scatter_mapbox(
            df_sub, lat="latitude", 
            lon="longitude", 
            hover_data=["price", "median_income","num_amenities","review_scores_rating"],
            hover_name="id",
            color_discrete_sequence=["#F66F9F"], 
            zoom=7, height=650)
    fig1.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})

    if roomtype_choice=='All types':
        roomtype=df['log_price']
    else:
        roomtype=df['log_price'][df['room_type']==roomtype_choice]
    fig2 = ff.create_distplot([roomtype], group_labels=[roomtype_choice], bin_size=0.3,colors=['#FAAFCA'], show_rug=False)

    fig2.update_layout(title=f"<b>Distribution of log(Price) for {roomtype_choice}</b>")

    fig2.layout.plot_bgcolor = 'white'

    return fig1, fig2


@app.callback(Output('describe_stats_table', 'figure'),
              Input('describe_stats_drop_down', 'value'))
def update_describe_stats_table(value):
    fig = go.Figure(data=[go.Table(
        header=dict(values=['<b>Statistics</b>', '<b>Variable</b>'],
                    line_color='darkslategray',
                    fill_color='pink',
                    align='center',
                   font_size=15,height = 45),
        cells=dict(values=[stat.index, # 1st column
                           round(stat[value],2)], # 2nd column
                   line_color='darkslategray',
#                    fill_color='lightcyan',
                   align='center',
                   height = 40,
                  font_size=15))
    ])

    fig.update_layout(height=570)
    return fig


@app.callback(
    Output('corr_plot', 'figure'),
    Input('measurement', 'value'),
    )
def update_graph(measurement):
    df_corr_r = df[df_b]
    df_corr_round = df_corr_r.corr()[[measurement]].T[df_a].T.round(2)
    colorscale=[[0.0,'rgb(255,255,255)'],[1.0,'rgb(203,105,140)']]
    fig_cor = ff.create_annotated_heatmap(
        z=df_corr_round.to_numpy(),
        x=df_corr_round.columns.tolist(),
        y=df_corr_round.index.tolist(),
        zmax=1,
        zmin=-1,
        showscale=True,
        hoverongaps=True,
        colorscale=colorscale
    )
    fig_cor.update_layout(
        margin={'l': 0, 'b': 20, 't': 30, 'r': 0},
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right")
    )
    fig_cor.update_layout(yaxis_tickangle=0)
    fig_cor.update_layout(xaxis_tickangle=0)
    fig_cor.update_layout(title_text="", height=270)
    return fig_cor


@app.callback(Output('radar_plot', 'figure'),
              Input('scatter_output', 'hoverData'))
def update_rader_plot(hoverData):
    id_num = hoverData['points'][0]['customdata']
    ddf = pd.DataFrame(dict(
        r=df[df['id']==id_num][['review_scores_accuracy', 'review_scores_checkin','review_scores_cleanliness', 'review_scores_communication','review_scores_location', 'review_scores_value']].values[0],
        theta=['Accuracy','Cleanliness','Checkin',
               'Communication', 'Location', 'Value']))
    fig = px.line_polar(ddf, r='r', theta='theta', line_close=True, height = 300)
    fig.update_traces(fill='toself',line_color='#CB698C')
    fig.update_layout(
          
          polar=dict(
            radialaxis=dict(
              visible=True,
              range=[2.5, 5])),
          title={
            'text': f"<b>Rating Details of House ID = {id_num}</b>",
            'y':0.97,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            titlefont= {"size": 20},
            title_font_family='Arial',
            margin={'l': 30, 'b': 30, 't': 70, 'r': 30},
            showlegend=False,
            
            )
    return fig


@app.callback(Output('scatter_output', 'figure'),
              Input('x_dropdown', 'value'),)
def update_scatter_plot(x_val):
    fig = px.scatter(df, x = x_val, y = df['log_price'], trendline="ols", hover_name = 'id')
    fig.data[1].line.color = 'black'
    fig.update_layout(hovermode='closest', height = 570,
                    plot_bgcolor='white',
                    xaxis=dict(
                                showline=True,
                                showgrid=True,
                                zeroline=True,
                                showticklabels=True,
                                linecolor='rgb(233,233,233)',
                                linewidth=2,
                                           ),
                    yaxis=dict(
                                showgrid=True,
                                zeroline=True,
                                showline=True,
                                showticklabels=True,
                                linecolor='rgb(233,233,233)',
                                        ),
                     margin={'l': 0, 'b': 0, 't': 30, 'r': 0}
                     )
    fig.update_traces(customdata=df['id'], 
                      marker_color='rgba(203,105,140, 0.8)'
                     )
    return fig

# callback function
@app.callback(
    Output("test_output", "children"),
    [Input("room_type-input", "value"),
    Input('rs_rating_input','value'),
    Input('number_of_reviews','value'),
    Input('beds','value'),
    Input('bedrooms','value'),
    Input('median_income','value'),
    Input('h_t_l_count','value'),
    Input('host_is_superhost','value'),
    Input('minimum_nights','value'),
    Input('availability_365','value'),
    Input("select-1", "value"),
    Input("select-2", "value")
    ],
)
def lr_pred_(room_type,rs_rating_input,number_of_reviews,beds,bedrooms,median_income,h_t_l_count,host_is_superhost,minimum_nights,availability_365,amenities_1,amenities_2):
    # return f'you select {room_type},{rs_rating_input},{number_of_reviews},{beds},{bedrooms},{median_income},{h_t_l_count},{host_is_superhost},{minimum_nights},{availability_365}'
    # dummy calculation
    v1,v2 = 0,0
    if room_type == 'Private room':
        v1 = 1
        v2 = 0
    elif room_type == 'Shared room':
        v1 = 0
        v2 = 1
    else:
        v1 = 0
        v2 = 0
    # interaction calculation
    inter1 = rs_rating_input*number_of_reviews
    # log calculation
    log_median_income = math.log(median_income)
    # num_amenity calculation
    num_amenities = len(amenities_1+amenities_2)
    lr_pred = lr_default(v1,v2,rs_rating_input,number_of_reviews,inter1,beds,bedrooms,log_median_income,h_t_l_count,host_is_superhost,minimum_nights,num_amenities,availability_365)
    return f'The expected value for the log price is {round(lr_pred,4)}, actual price is {round(math.exp(lr_pred),4)}'

@app.callback(
    Output("test_output-2", "children"),
    [Input("room_type-input", "value"),
    Input('rs_rating_input','value'),
    Input('number_of_reviews','value'),
    Input('beds','value'),
    Input('bedrooms','value'),
    Input('median_income','value'),
    Input('h_t_l_count','value'),
    Input('host_is_superhost','value'),
    Input('minimum_nights','value'),
    Input('availability_365','value'),
    Input("select-1", "value"),
    Input("select-2", "value")
    ],
)
def ci_cal_(room_type,rs_rating_input,number_of_reviews,beds,bedrooms,median_income,h_t_l_count,host_is_superhost,minimum_nights,availability_365,amenities_1,amenities_2):
    # dummy calculation
    v1,v2 = 0,0
    if room_type == 'Private room':
        v1 = 1
        v2 = 0
    elif room_type == 'Shared room':
        v1 = 0
        v2 = 1
    else:
        v1 = 0
        v2 = 0
    # interaction calculation
    inter1 = rs_rating_input*number_of_reviews
    # log calculation
    log_median_income = math.log(median_income)
    # num_amenity calculation
    num_amenities = len(amenities_1+amenities_2)
    lr_pred = lr_default(v1,v2,rs_rating_input,number_of_reviews,inter1,beds,bedrooms,log_median_income,h_t_l_count,host_is_superhost,minimum_nights,num_amenities,availability_365)
    ci = ci_cal(lr_pred)
    return f'The confidence interval of the price is [{round(math.exp(ci[0]),4)},{round(math.exp(ci[1]),4)}]'

@app.callback(
    dash.dependencies.Output('model-output', 'figure'),
    [dash.dependencies.Input("room_type-input", "value"),
    dash.dependencies.Input("rs_rating_input", "value"),
    dash.dependencies.Input("number_of_reviews", "value"),
    dash.dependencies.Input("beds", "value"),
    dash.dependencies.Input("bedrooms", "value"),
    dash.dependencies.Input("median_income", "value"),
    dash.dependencies.Input("h_t_l_count", "value"),
    dash.dependencies.Input("host_is_superhost", "value"),
    dash.dependencies.Input("minimum_nights", "value"),
    dash.dependencies.Input("availability_365", "value"),
    dash.dependencies.Input("select-1", "value"),
    dash.dependencies.Input("select-2", "value")
    ]
)
def update_attribute_graph(room_type,rs_rating_input,number_of_reviews,beds,bedrooms,median_income,h_t_l_count,host_is_superhost,minimum_nights,availability_365,amenities_1,amenities_2):
    v1,v2 = 0,0
    if room_type == 'Private room':
        v1 = 1
        v2 = 0
    elif room_type == 'Shared room':
        v1 = 0
        v2 = 1
    else:
        v1 = 0
        v2 = 0
    # interaction calculation
    inter1 = rs_rating_input*number_of_reviews
    # log calculation
    log_median_income = math.log(median_income)
    # num_amenity calculation
    num_amenities = len(amenities_1+amenities_2)
    lr_pred = lr_default(v1,v2,rs_rating_input,number_of_reviews,inter1,beds,bedrooms,log_median_income,h_t_l_count,host_is_superhost,minimum_nights,num_amenities,availability_365)
    ci = ci_cal(lr_pred)
    graph_data = data_[(data_['log_price']>=ci[0])&(data_['log_price']<=ci[1])].copy()
    # annotations = []
    
    # our plotly here
    fig = make_subplots(rows=4, cols=2,subplot_titles=('review_scores_rating','number_of_reviews','beds','bedrooms','median_income','host_total_listings_count','minimum_nights','availability_365'))
    fig.add_trace(go.Box(
        x=graph_data['review_scores_rating'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3,
        name='',
        marker_color = 'rgb(252,185,207)',
        showlegend = False
        # marker_color='#3D0970'
    ), row=1, col=1)
    fig.add_vline(x=rs_rating_input, line_width=3, line_dash="dash", line_color="pink",row=1, col=1,annotation_text=f"{rs_rating_input}", annotation_position="top right")


    fig.add_trace(go.Box(
        x=graph_data['number_of_reviews'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3,
        name='',
        marker_color = 'rgb(199,225,252)',
        showlegend = False
    ), row=1, col=2)
    fig.add_vline(x=number_of_reviews, line_width=3, line_dash="dash", line_color="pink",row=1, col=2,annotation_text=f"{number_of_reviews}", annotation_position="top right")


    fig.add_trace(go.Box(
        x=graph_data['beds'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3,
        name='',
        marker_color = 'rgb(252,185,207)',
        showlegend = False
    ), row=2, col=1)
    fig.add_vline(x=beds, line_width=3, line_dash="dash", line_color="pink", row=2, col=1,annotation_text=f"{beds}", annotation_position="top right")


    fig.add_trace(go.Box(
        x=graph_data['bedrooms'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3,
        name='',
        marker_color = 'rgb(199,225,252)',
        showlegend = False
    ), row=2, col=2)
    fig.add_vline(x=bedrooms, line_width=3, line_dash="dash", line_color="pink", row=2, col=2,annotation_text=f"{bedrooms}", annotation_position="top right")


    fig.add_trace(go.Box(
        x=graph_data['median_income'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3,
        name='',
        marker_color = 'rgb(252,185,207)',
        showlegend = False
    ), row=3, col=1)
    fig.add_vline(x=median_income, line_width=3, line_dash="dash", line_color="pink", row=3, col=1,annotation_text=f"{median_income}", annotation_position="top right")


    fig.add_trace(go.Box(
        x=graph_data['host_total_listings_count'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3,
        name='',
        marker_color = 'rgb(199,225,252)',
        showlegend = False
    ), row=3, col=2)
    fig.add_vline(x=h_t_l_count, line_width=3, line_dash="dash", line_color="pink", row=3, col=2,annotation_text=f"{h_t_l_count}", annotation_position="top right")


    fig.add_trace(go.Box(
        x=graph_data['minimum_nights'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3,
        name='',
        marker_color = 'rgb(252,185,207)',
        showlegend = False
    ), row=4, col=1)
    fig.add_vline(x=minimum_nights, line_width=3, line_dash="dash", line_color="pink", row=4, col=1,annotation_text=f"{minimum_nights}", annotation_position="top right")


    fig.add_trace(go.Box(
        x=graph_data['availability_365'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3,
        name='',
        marker_color = 'rgb(199,225,252)',
        showlegend = False,
        # xaxis=dict(showgrid = False)
    ), row=4, col=2)
    fig.add_vline(x=availability_365, line_width=3, line_dash="dash", line_color="pink", row=4, col=2,annotation_text=f"{availability_365}", annotation_position="top right")

    fig.update_layout(
                    paper_bgcolor='rgb(252, 252, 255)',
                    plot_bgcolor='rgb(248, 248, 255)',
                    title_text="General Features and Attributes",
                    # annotations = annotations, 
                    showlegend = False                  
    )    

    # Update xaxis properties
    for i in range(1,5):
        for j in range(1,3):
            fig.update_xaxes(showgrid=False, row=i, col=j)

    return fig

@app.callback(
    dash.dependencies.Output('model-output-2', 'figure'),
    [dash.dependencies.Input("room_type-input", "value"),
    dash.dependencies.Input("rs_rating_input", "value"),
    dash.dependencies.Input("number_of_reviews", "value"),
    dash.dependencies.Input("beds", "value"),
    dash.dependencies.Input("bedrooms", "value"),
    dash.dependencies.Input("median_income", "value"),
    dash.dependencies.Input("h_t_l_count", "value"),
    dash.dependencies.Input("host_is_superhost", "value"),
    dash.dependencies.Input("minimum_nights", "value"),
    dash.dependencies.Input("availability_365", "value"),
    dash.dependencies.Input("select-1", "value"),
    dash.dependencies.Input("select-2", "value")
    ]
)
def update_attribute_graph2(room_type,rs_rating_input,number_of_reviews,beds,bedrooms,median_income,h_t_l_count,host_is_superhost,minimum_nights,availability_365,amenities_1,amenities_2):
    v1,v2 = 0,0
    if room_type == 'Private room':
        v1 = 1
        v2 = 0
    elif room_type == 'Shared room':
        v1 = 0
        v2 = 1
    else:
        v1 = 0
        v2 = 0
    # interaction calculation
    inter1 = rs_rating_input*number_of_reviews
    # log calculation
    log_median_income = math.log(median_income)
    # num_amenity calculation
    num_amenities = len(amenities_1+amenities_2)
    lr_pred = lr_default(v1,v2,rs_rating_input,number_of_reviews,inter1,beds,bedrooms,log_median_income,h_t_l_count,host_is_superhost,minimum_nights,num_amenities,availability_365)
    ci = ci_cal(lr_pred)
    graph_data = data_[(data_['log_price']>=ci[0])&(data_['log_price']<=ci[1])].copy()
    annotations = []
    # room_type_ = list(graph_data['room_type'].unique())
    # host_is_superhost_ = list(graph_data['host_is_superhost'].unique())
    # color_ = []
    # for room,host,color in zip(room_type_,host_is_superhost_,color_):
    #     fig = px.box(graph_data, x="room_type", y="price",points='all',color ='host_is_superhost')
    # fig.update_layout(title_text="Price Boxplot for Private & Shared Room among those selected houses")    
    fig = px.box(graph_data, x="room_type", y="price",points='all',color ='host_is_superhost',color_discrete_sequence = ['rgb(252,206,221)','rgb(199,225,252)'])
    annotations.append(dict(xref='paper', yref='paper', x=-0.05, y=1.1,
                                xanchor='left', yanchor='bottom',
                                text='Price Boxplot for Private & Shared Room among those selected houses',
                                font=dict(family='Arial',
                                            size=20,
                                            color='rgb(72,72,72)'),
                                showarrow=False))

    fig.update_layout(
                    paper_bgcolor='rgb(252, 252, 255)',
                    plot_bgcolor='rgb(248, 248, 255)',
                    annotations = annotations,                   
    )    


    return fig


if __name__ == '__main__':
    app.run_server(debug=True)