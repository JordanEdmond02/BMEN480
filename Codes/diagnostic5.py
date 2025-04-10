import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State, dash_table
import os
import zipfile
import io
import numpy as np
import base64
import pandas as pd
import pydicom as dicom
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage import measure, morphology, feature
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import trimesh
from scipy.ndimage import gaussian_filter, binary_fill_holes

# Logos
NCAT_Logo = "assets/NCATLogo.png"
JNJ_Logo = "assets/jjmt_logo.png"

# File paths
UPLOAD_FOLDER = './temp_uploads'
OUTPUT_FOLDER = './temp_outputs'

#Colors
ncat_blue =  "#36618e"
ncat_gold = "#F1C40F"
light_grey = "#EAEDED"

# Header
header = dbc.Navbar(
    dbc.Container(
        dbc.Row([
            dbc.Col(html.Img(src=NCAT_Logo, style={"height": "75px"}), width="auto"),
            dbc.Col(html.Img(src=JNJ_Logo, style={"height": "85px"}), width="auto"),
            dbc.Col(html.H1("Bone-AFide-Scanners", style={"textAlign": "center", "font-size": "275%", "margin": "0"})),
        ]),
    ),
    color= light_grey,
)
# Download Row - Single Horizontal Row
download_row = dbc.Container(
    dbc.Row([
        dbc.Col(
            dcc.Upload(
                id="upload-data",
                children=dbc.Button("Upload DICOM Folder (.zip)",style={"backgroundColor": "#36618e", "color": "white"}),
                multiple=False,
                style={
                    "width": "100%",
                    "height": "40px",
                    "lineHeight": "40px",
                    "textAlign": "center",
                    "margin": "10px auto",
                },
            ),
            width="auto"
        ),
        dbc.Col(
            dcc.Input(
                id="filename-input",
                type="text",
                placeholder="Enter STL filename",
                debounce=True,
                style={"width": "100%"}
            ),
            width=3
        ),
        #STL Button
        dbc.Col(
            dbc.Button('Download STL', id='download-button', style={"backgroundColor": "#36618e", "color": "white"},  n_clicks=0),
            width="auto"
        ),
        dbc.Col(
            html.Div(id="save-status", style={"font-weight": "bold"}),
            width="auto"
        ),
        dcc.Download(id="download-stl"),
    ],
    align="center",
    justify="between",
    style={"marginTop": "20px"}
    )
)

def clean_old_files(folderpath):
    """Removes old files from the temporary directory."""
    if os.path.exists(folderpath):
        for file in os.listdir(folderpath):
            file_path = os.path.join(folderpath, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def make_dicom_list(folderpath):
    dicom_list = []
    for file in os.listdir(folderpath):
        if file.endswith('.dcm'):
            try:
                dcm = dicom.dcmread(os.path.join(folderpath, file), force=True)
                if "TransferSyntaxUID" not in dcm.file_meta:
                    dcm.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
                if dcm.file_meta.TransferSyntaxUID.is_compressed:
                    dcm.decompress()
                dicom_list.append(dcm)
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return sorted(dicom_list, key=lambda x: int(getattr(x, 'InstanceNumber', 0)))

def create_array(dicom_index):
    slices = np.stack([slice.pixel_array for slice in dicom_index], axis=-1)
    return slices.astype(np.float32)

def segmentationprocess(image):
    threshold = threshold_otsu(image)
    binaryimage = image > threshold
    clearedbinary = clear_border(binaryimage)
    filteredbinary = remove_small_objects(clearedbinary, min_size=1000)  # Increased min size to remove small artifacts
    smoothedbinary = binary_closing(filteredbinary, footprint=disk(5))


    # Apply edge detection
   # canny_edges = feature.canny(smoothedbinary, sigma=1.5)
    #dilated_edges = morphology.dilation(canny_edges, morphology.disk(3))
   # filled_image = binary_fill_holes(dilated_edges)
   # eroded_image = morphology.erosion(filled_image, morphology.disk(3))

    return smoothedbinary

def segmentation_over_all(segmentedslices):
    return np.array([segmentationprocess(segmentedslices[:, :, i]) for i in range(segmentedslices.shape[2])]).transpose(
        1, 2, 0)

def extract_zip_contents(zip_file_content, output_folder):
    clean_old_files(output_folder)
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    with zipfile.ZipFile(io.BytesIO(zip_file_content), "r") as zip_ref:
        zip_ref.extractall(output_folder)
    return output_folder

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    header,
    download_row,
    dcc.Store(id="stored-points", data=[]),
    html.H3("ROI"),
    html.Div(id='point-location'),
    dash_table.DataTable(
        id='point-table',
        columns=[
            {"name": "X", "id": "X"},
            {"name": "Y", "id": "Y"},
            {"name": "Z", "id": "Z"},
            {"name": "Notes (Press Enter to Save)", "id": "notes", "editable": True},
        ],
        data=[],  # Initially empty
        export_format = "xlsx",
        editable=True,
        row_deletable=True
    ),
    html.Div(id="plotly-view"), #3D output
])

#ZIP File to 3D Model Callback
@app.callback(
    Output("plotly-view", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def plotly_view(contents, filename):
    if not contents:
        return html.Div("Error: No file uploaded.")

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    if not filename.endswith(".zip"):
        return html.Div("Error: Please upload a ZIP file containing DICOM images.")

    upload_path = UPLOAD_FOLDER
    folderpath = extract_zip_contents(decoded, upload_path)
    dicom_files = make_dicom_list(folderpath)

    if not dicom_files:
        return html.Div("Error: No valid DICOM files found.")

    slices_array = create_array(dicom_files)
    smoothed_slices = gaussian_filter(slices_array, sigma=2.0)  # Increased smoothing to reduce noise
    segmented_image_array = segmentation_over_all(smoothed_slices)
    verts, faces, _, _ = measure.marching_cubes(segmented_image_array, level=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)

    # Ensure the OUTPUT_FOLDER exists before saving
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure the output folder exists
    stl_file_path = os.path.join(OUTPUT_FOLDER, "bone_model.stl")
    mesh.export(stl_file_path)

    x, y, z = verts.T
    i, j, k = faces.T

    mesh_fig = go.Figure(
        data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightblue', opacity=0.5)]
    )
    mesh_fig.update_layout(title="3D Bone Model", height=700, width=800)

    return dcc.Graph(figure=mesh_fig, id = 'mesh-fig')

#3D point bank callback-click point, then it appends to table, and the user can write notes on the table for the point
@app.callback(
    Output("point-table", "data"),
    Input("mesh-fig", "clickData"),
    State("point-table",'data'),
    prevent_initial_call=True,
)
def update_table(click_data,table):
    if click_data is None:
        return table
    if table is None:
        table=[]
    point = click_data['points'][0] #extracts first point,
    new_row = {"X": point['x'], "Y": point['y'], "Z": point['z']} #extracts x,y,z coordinates of clicked point
    table.append(new_row)
    return table

#STL Download Callback
@app.callback(
    Output("download-stl", "data"),
    Input("download-button", "n_clicks"),
    State("filename-input", "value"),
    prevent_initial_call=True,
)
def download_stl(n_clicks, filename):
    if n_clicks > 0:
        # Ensure the output folder exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure the output folder exists

        # Default or user-defined filename for download
        if filename:
            stl_file_path = os.path.join(OUTPUT_FOLDER, "bone_model.stl")
            download_filename = f"{filename}.stl"
        else:
            stl_file_path = os.path.join(OUTPUT_FOLDER, "bone_model.stl")
            download_filename = "bone_model.stl"

        return dcc.send_file(stl_file_path, download_filename)



if __name__ == "__main__":
    app.run(debug=True)
