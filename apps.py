import datetime
import base64
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dropbox
import cv2
from matplotlib import pyplot as plt
from urllib.request import urlopen
import numpy as np
import pandas as pd
dbx = dropbox.Dropbox('5uSdWA0gd2UAAAAAAAAAAauPVaO_t_nlwRgP3YzwZ8-2HlxYFWRLUrmTAgk4F4b7')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server
app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
     html.Div([
     html.Button('Start Process', id='btn-nclicks-1', n_clicks=0)]),
     html.Div(id='container-button-timestamp'),
])

def parse_contents(contents, filename, date):
    dbx.files_delete_v2('/IMAGE.png')
    content_type, content_string = contents.split(',')
    base64_img_bytes = content_string.encode('utf-8')
##    with open('decoded_image.png', 'wb') as file_to_save:
    decoded_image_data = base64.decodebytes(base64_img_bytes)
        #file_to_save.write(decoded_image_data,format="PNG")
    dbx.files_upload(
                decoded_image_data,'/IMAGE.png', dropbox.files.WriteMode.add,
                mute=True)
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,width="200",height="200"),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
@app.callback(Output('container-button-timestamp', 'children'),
              Input('btn-nclicks-1', 'n_clicks'))
def displayClick(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        msg=''
        for entry in dbx.files_list_folder('').entries:
            aa=entry.name
            if aa=='IMAGE.png':
                bb=entry.id
                resultresult =dbx.files_get_temporary_link(bb)
                cc=resultresult.link
        url_response = urlopen(cc)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
        h, w, c = img.shape
        y=int(h*16/100)
        x=int(w*16/100)
        crop_image = img[x:w,y:h]
        grayimg = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        Gaussian_Iamge = cv2.GaussianBlur(grayimg,(5,5),0)
        edges = cv2.Canny(Gaussian_Iamge,30,30)
        df = pd.DataFrame(edges/255)
        Descibe_Data=df.describe()
        df1=Descibe_Data.mean(axis=1)
        med1=df1.iloc[1:len(df1)].mean(axis=0)
        df2=Descibe_Data.median(axis=1)
        med2=df2.iloc[1:len(df2)].mean(axis=0)
        df3=Descibe_Data.std(axis=1)
        med3=df3.iloc[1:len(df3)].mean(axis=0)
        df4=pd.read_csv('TB.csv')
        df5=pd.read_csv('Covid.csv')
        df6=pd.read_csv('PE.csv')
        data_point_TB=np.array([df4['Mean'],df4['Median']])
        data_point_Covid=np.array([df5['Mean'],df5['Median']])
        data_point_PE=np.array([df6['Mean'],df6['Median']])
        daat_point_test_image=np.array([round(med1,4),round(med2,4)])
        Euclidean_distance_TB = round(np.linalg.norm(data_point_TB - daat_point_test_image),4)
        Euclidean_distance_Covid = round(np.linalg.norm(data_point_Covid - daat_point_test_image),4)
        Euclidean_distance_PE = round(np.linalg.norm(data_point_PE - daat_point_test_image),4)
        list9=[str("TB"),str("Covid"),str("PE")]
        list8=[Euclidean_distance_TB, Euclidean_distance_Covid,Euclidean_distance_PE]
        new_data=pd.DataFrame({'TB' : [list8[0]],
                                   'Covid' : [list8[1]],
                                         "PE" : [list8[2]]}, 
                                  columns=['TB', 'Covid','PE'])
        data = new_data.to_csv(index=False) # The index parameter is optional
        with io.BytesIO(data.encode()) as stream:
            stream.seek(0)
            dbx.files_upload(stream.read(), "/Chest X-ray.csv", mode=dropbox.files.WriteMode.overwrite)
    else:
        msg = ''
    return html.Div(msg)

if __name__ == '__main__':
    app.run_server(debug=False)
