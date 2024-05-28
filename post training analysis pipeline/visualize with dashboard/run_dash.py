# pip3 install flask_httpauth plotly dash

import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from flask import Flask, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from PIL import Image
import base64
from io import BytesIO
import io
import imageio

# Number of items per page
ITEMS_PER_PAGE = 250
BAR_PLOT_HEIGHT = 500

def numpy_array_to_image_string(frame):
    frame = frame.transpose(1,2,0)
    img_fluorescence = frame[:,:,[2,1,0]]
    img_dpc = frame[:,:,3]
    img_dpc = np.dstack([img_dpc,img_dpc,img_dpc])
    img_overlay = 0.64*img_fluorescence + 0.36*img_dpc
    frame = img_overlay.astype('uint8')
    #frame = np.hstack([img_dpc,img_fluorescence,img_overlay]).astype('uint8')
    img = Image.fromarray(frame, 'RGB')
    with io.BytesIO() as buffer:
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()


# Initialize the Flask app
server = Flask(__name__)
auth = HTTPBasicAuth()

# User credentials
users = {
    "cephla": generate_password_hash("octopi")
}

@auth.verify_password
def verify_password(username, password):
    if username in users:
        return check_password_hash(users.get(username), password)
    return False

# Middleware to protect Dash routes
@server.before_request
def before_request_func():
    # Authenticate all non-static requests
    if not request.endpoint or request.endpoint == 'static':
        return
    return auth.login_required(lambda: None)()

# Initialize the Dash app
app = dash.Dash(__name__, server=server)

# Generating threshold values
threshold_values = np.arange(0.95, 1.001, 0.001)

# Loading the datasets
thresholds = {f"{threshold:.3f}": pd.read_csv(f'count vs threshold/all_dataset_prediction_counts_{threshold:.3f}.csv') 
              for threshold in threshold_values}

# App layout
app.layout = html.Div([
    dcc.Graph(id='main-plot'),
    html.Div([
        html.Label('Threshold', style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='threshold-dropdown',
            options=[{'label': k, 'value': k} for k in thresholds.keys()],
            style={'width': '100px'},
            clearable=False,
            value='0.970'  # Default value
        )
    ], style={'display': 'flex','justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '20px'}),
    html.H4(id='sample-id', style={'marginTop': '10px','marginBottom': '0px','textAlign': 'center'}),
    dcc.Graph(id='secondary-plot',style={'marginTop': '0px'}),
    # html.Div(id='image-grid'),
    dcc.Loading(
        id="loading",
        children=[html.Div(id='image-grid')],
        type="cube",  # You can choose from 'graph', 'cube', 'circle', 'dot', or 'default'
    ),
    html.Div([
        "Page: ",
        dcc.Input(id='page-input', type='number', value=1, min=1, style={'width': '100px'}),
    ])

])

# Callback to update main plot based on selected threshold
@app.callback(
    Output('main-plot', 'figure'),
    [Input('threshold-dropdown', 'value')]
)
def update_main_plot(selected_threshold):
    df = thresholds[selected_threshold]
    df['Sample ID'] = df['dataset ID'].str.extract(r'(.*?)_\d{4}-\d{2}-\d{2}')
    # fig = px.bar(df, x='dataset ID', y='Positives per 5M RBC', log_y=True)
    fig = px.bar(df, x='Sample ID', y='Positives per 5M RBC', hover_data=['dataset ID'], log_y=True)
    fig.update_layout(height=BAR_PLOT_HEIGHT)
    fig.add_hline(y=5, line_dash="dash", line_color="red")
    return fig

# Callback to update secondary plot based on click data
@app.callback(
    Output('secondary-plot', 'figure'),
    [Input('main-plot', 'clickData')],
    [State('threshold-dropdown', 'value')]
)
def update_secondary_plot(clickData, selected_threshold):
    if clickData is None:
        return px.scatter()

    df = thresholds[selected_threshold]
    # dataset_id = clickData['points'][0]['x']
    dataset_id = clickData['points'][0]['customdata'][0]
    pos_per_5m_rbc = []
    for threshold, df in thresholds.items():
        data = df[df['dataset ID'] == dataset_id]
        if not data.empty:
            pos_per_5m_rbc.append((float(threshold), data['Positives per 5M RBC'].values[0]))
    threshold_df = pd.DataFrame(pos_per_5m_rbc, columns=['Threshold', 'Positives per 5M RBC'])
    new_fig = px.line(threshold_df, x='Threshold', y='Positives per 5M RBC', markers=True)
    return new_fig


# Callback to update images
@app.callback(
    Output('image-grid', 'children'),
    [Input('main-plot', 'clickData'),
     Input('page-input', 'value')]
)
def update_images(clickData, page_number):

    if clickData is not None:

        # print(clickData)
        # dataset_id = clickData['points'][0]['x']
        dataset_id = clickData['points'][0]['customdata'][0]
        selected_file = '/home/octopi/Desktop/Octopi/data/npy_v2/' + dataset_id + '.npy'

        npy_data = np.load(selected_file)
        csv_data = pd.read_csv('../model output/' + dataset_id + '.csv')

        index = csv_data['index']
        sorted_scores = csv_data['parasite output']
        sorted_data = npy_data[index]

        # Pagination logic
        start_index = (page_number - 1) * ITEMS_PER_PAGE
        end_index = start_index + ITEMS_PER_PAGE
        paginated_data = sorted_data[start_index:end_index]
        paginated_scores = sorted_scores[start_index:end_index]

        # Convert numpy arrays to images and encode them
        image_elements = []
        for i, (arr, score) in enumerate(zip(paginated_data, paginated_scores), start=start_index + 1):
            encoded_image = numpy_array_to_image_string(arr)
            image_html = html.Img(src=f"data:image/png;base64,{encoded_image}", style={'height':'124px', 'width':'124px'})
            score_html = html.Div(f"{score:.2f}", style={'textAlign': 'center'})
            number_html = html.Div(f"[{i}]", style={'textAlign': 'center', 'color': 'gray'})
            image_elements.append(html.Div([number_html, image_html, score_html], style={'margin':'10px', 'display':'inline-block'}))

        return html.Div(image_elements)

    return html.Div()


# Callback to reset page number when a new file is selected
@app.callback(
    Output('page-input', 'value'),
    [Input('main-plot', 'clickData')],
    [State('page-input', 'value')]
)
def reset_page_number(clickData, current_page):
    return 1 if clickData else current_page

# Callback to update selected file name
@app.callback(
    Output('sample-id', 'children'),
    [Input('main-plot', 'clickData')]
)
def update_file_name(clickData):
    # return f"Sample: {clickData['points'][0]['x']}" if clickData else "No sample selected"
    return f"Sample: {clickData['points'][0]['customdata'][0]}" if clickData else "No sample selected"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port = 8056)
