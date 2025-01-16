import dash_bootstrap_components as dbc
import dash
from dash import Dash, html, dcc, callback, Output, Input, State
from dash import dcc
import pandas as pd
import numpy as np
import plotly.express as px
import pydicom as dicom
import matplotlib.pyplot as pl
import base64
import pybase64
import io
import dicom2jpg
import plotly.graph_objs as go
import PIL.Image
from io import BytesIO

# Initialize the app
app = Dash(__name__)

# Global variables to hold the images and grid data
uploaded_images = []
x_vals, y_vals, z_vals = [], [], []  # 3D grid values

# Function to generate 3D grid data
def generate_3d_grid(x_points, y_points, z_points):
    x = np.linspace(-5, 5, x_points)
    y = np.linspace(-5, 5, y_points)
    z = np.linspace(-5, 5, z_points)

    # Create a mesh grid from x, y, z values
    X, Y, Z = np.meshgrid(x, y, z)

    # Flatten the grid for scatter3d input
    x_vals = X.flatten()
    y_vals = Y.flatten()
    z_vals = Z.flatten()

    return x_vals, y_vals, z_vals

# Generate initial 3D grid with 10 points along each axis
x_vals, y_vals, z_vals = generate_3d_grid(10, 10, 10)

# Define the layout and the 3D scatter plot
app.layout = html.Div(children=[
    html.H1('3D Grid Visualization and Image Slider'),

    # Slider to control the number of points in the grid
    html.Div([
        html.Label("Adjust Number of Points in the Grid:"),
        dcc.Slider(
            id='point-slider',
            min=5,  # Minimum number of points
            max=50,  # Maximum number of points
            step=1,  # Step size
            value=10,  # Initial value
            marks={i: str(i) for i in range(5, 51, 5)},
        ),
    ], style={'padding': '20px'}),

    # File upload area
    html.Div([
        html.Label("Upload Images to Flip Through (optional):"),
        dcc.Upload(
            id='upload-images',
            children=html.Button('Upload Images'),
            multiple=True  # Allow multiple files to be uploaded
        ),
        html.Div(id='output-data-upload'),
    ], style={'padding': '20px'}),

    # Graph component to render the 3D scatter plot
    dcc.Graph(
        id='3d-grid',
        figure={
            'data': [
                go.Scatter3d(
                    x=x_vals,  # x values
                    y=y_vals,  # y values
                    z=z_vals,  # z values
                    mode='markers',  # Marker mode (points)
                    marker=dict(size=4, color='blue', opacity=0.8)
                ),
            ],
            'layout': go.Layout(
                title="3D Grid Visualization",
                scene=dict(
                    xaxis=dict(title='X Axis'),
                    yaxis=dict(title='Y Axis'),
                    zaxis=dict(title='Z Axis')
                ),
                margin=dict(l=0, r=0, b=0, t=40)  # Adjust margins for better display
            )
        }
    ),

    # Slider to flip through uploaded images
    html.Div([
        html.Label("Select Image Index:"),
        dcc.Slider(
            id='image-slider',
            min=0,  # The first image
            max=0,  # Will be updated dynamically based on uploaded images
            step=1,
            value=0,  # Start by displaying the first image
            marks={0: '0'},
        ),
    ], style={'padding': '20px'}),

    # Placeholder for displaying the image
    html.Div(id='image-display', children=[
        html.Img(id='current-image', style={'width': '100%', 'height': 'auto'})
    ])
])

# Callback to handle file uploads and update slider
@app.callback(
    [Output('image-slider', 'max'),
     Output('image-slider', 'marks'),
     Output('image-display', 'children')],
    [Input('upload-images', 'contents')],
    prevent_initial_call=True
)
def handle_file_upload(uploaded_files):
    global uploaded_images

    # If files are uploaded, process them
    if uploaded_files is not None:
        uploaded_images = []

        # Process each uploaded image
        for file in uploaded_files:
            content_type, content_string = file.split(',')
            decoded = base64.b64decode(content_string)
            try:
                # Convert the image data to an image object using PIL
                image = PIL.Image.open(BytesIO(decoded))

                # Convert image to base64 and append to the list
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                uploaded_images.append(img_str)
            except Exception as e:
                return f"Error processing file: {e}"

    # Update the slider based on the number of uploaded images
    marks = {i: str(i) for i in range(len(uploaded_images))}
    return len(uploaded_images) - 1, marks, html.Img(
        id='current-image',
        src=f"data:image/png;base64,{uploaded_images[0]}",  # Display the first image initially
        style={'width': '100%', 'height': 'auto'}
    )

# Callback to update the displayed image based on the slider value
@app.callback(
    Output('current-image', 'src'),
    [Input('image-slider', 'value')],
    prevent_initial_call=True
)
def update_image(slider_value):
    if uploaded_images:
        return f"data:image/png;base64,{uploaded_images[slider_value]}"
    return ''

# Callback to update the 3D grid based on the slider value
@app.callback(
    Output('3d-grid', 'figure'),
    [Input('point-slider', 'value')],
    prevent_initial_call=True
)
def update_graph(slider_value):
    # Generate a new grid based on slider value
    x_vals, y_vals, z_vals = generate_3d_grid(slider_value, slider_value, slider_value)

    # Return the updated figure
    return {
        'data': [
            go.Scatter3d(
                x=x_vals,  # x values
                y=y_vals,  # y values
                z=z_vals,  # z values
                mode='markers',  # Marker mode (points)
                marker=dict(size=4, color='blue', opacity=0.8)
            ),
        ],
        'layout': go.Layout(
            title=f"3D Grid Visualization ({slider_value} points per axis)",
            scene=dict(
                xaxis=dict(title='X Axis'),
                yaxis=dict(title='Y Axis'),
                zaxis=dict(title='Z Axis')
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    }

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
