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
import dash_table
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
col=['TB','Covid','PE']
dbx = dropbox.Dropbox('5uSdWA0gd2UAAAAAAAAAAauPVaO_t_nlwRgP3YzwZ8-2HlxYFWRLUrmTAgk4F4b7')
for entry in dbx.files_list_folder('').entries:
            aa=entry.name
            if aa=='Chest X-ray.csv':
                dbx.files_delete_v2('/Chest X-ray.csv')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colors = {
    'background': '#1f152e',
    'text': '#7FDBFF'
}
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Div([
                    html.H1("Acculi Labs CHEST X-ray SCREENING SYSTEM",style={'font-size': '50px','font-family':'Times New Roman','color':'orange'}),
                    html.P("R.R. Nagar,Banglore,560098,Karnataka,India",style={'font-size': '20px','font-family':'Times New Roman','color':'orange'}),
                         ], 
                    style = {'padding' : '50px',
                             'textAlign': 'center'}),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '99%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'color':'white',
            'backgroundColor': '#00308F',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
     html.Div([
     html.Button('Start Process', id='btn-nclicks-1', style={'backgroundColor': '#00308F',
                                                             'color':'white','width':'15%', 'border':'1px black solid',
                                                             'height': '40px','textalign':'center', 'marginLeft': '2px', 'marginTop': 0,
                                                             'font-size': '15px','font-family':'Courier New','borderStyle': 'groove'},n_clicks=0),
     html.Button('Show Process Data', id='btn-nclicks-2',style={'backgroundColor': '#00308F',
                                                             'color':'white','width':'15%', 'border':'1px black solid',
                                                             'height': '40px','textalign':'center', 'marginLeft': '2px', 'marginTop':0,
                                                             'font-size': '15px','font-family':'Courier New','borderStyle': 'groove'},n_clicks=0)]),
      dcc.ConfirmDialog(
        id='confirm',
        message='Process is goin on',
    ),
     html.Div(id='container-button-timestamp'),
    dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in col],
    style_header={'backgroundColor': '#8C4500'},
                           style_cell={
                               'backgroundColor': '#DC143C',
                                'color': 'white',
                                 'textAlign': 'center'}),
    html.Div([
    html.Div([dcc.Graph(
        id='plot',style={'backgroundColor': '#00308F'})],className='six columns'),
    html.Hr(),
    html.Div([html.H2('Biological Vision Result and Doctor Validation',style={'font-size': '20px','font-family':'Times New Roman','color':'orange'}),
              html.H3('Note : Select Result (only for Doctors)',style={'font-size': '18px','font-family':'Times New Roman','color':'orange'}),
        dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'TB Covid PE', 'value': 'TB'},
            {'label': 'Covid TB PE', 'value': 'Covid'},
            {'label': 'PE Covid TB', 'value': 'PE'},
            {'label': 'TB PE Covid', 'Value': 'TB'},
            {'label': 'Covid PE TB', 'Value': 'Covid'},
            {'label': 'PE TB Covid', 'Value': 'PE'},
            
            
        ],
        placeholder="Select Answer",
                            style = dict(
                            width = '80%',
                            display = 'inline-block',
                            verticalAlign = "middle"
                            ))],
    className='six columns'),
     html.Div([html.Div(id='dd-output-container')],className='six columns'),
     html.Div([html.P('In the System when doctor select the result in this dropdown box,According to this result,It will be distributed according to rank system,the result with 1st rank, result will appera of that',
            style={'font-size': '18px','font-family':'Times New Roman','color':'orange','text-align':'justify'})])],
    className='row'),
    html.Br(),
    html.Div([
     html.Button('Download Report', id='btn-nclicks-3', style={'backgroundColor': '#00308F',
                                                             'color':'white','width':'100%', 'border':'2px white',
                                                             'height': '50px','textalign':'center', 'marginLeft': '2px', 'marginTop': 0,
                                                             'font-size': '20px','font-family':'Courier New','borderStyle': 'dashed'},n_clicks=0)]),
    html.Div(id='container-button'),
    
    
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
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,width="200",height="200"),
        html.H5("Test Image"),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Hr(),
        #html.Div('Raw Content'),
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
        msg='Process is Going on'
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
        percent_1=(list8[0])/(sum(list8))
        percent_2= (list8[1])/(sum(list8))
        percent_3 =(list8[2])/(sum(list8))
        list10=[round(percent_1*100,2),round(percent_2*100,2),round(percent_3*100,2)]
        new_data=pd.DataFrame({'TB' : [list10[0]],
                                   'Covid' : [list10[1]],
                                         "PE" : [list10[2]]}, 
                                  columns=['TB', 'Covid','PE'])
        data = new_data.to_csv(index=False) # The index parameter is optional
        with io.BytesIO(data.encode()) as stream:
            stream.seek(0)
            dbx.files_upload(stream.read(), "/Chest X-ray.csv", mode=dropbox.files.WriteMode.overwrite)
        msg="Process IS Complete"
    else:
        msg = 'Button has not been yet Clicked'
    return html.Div(msg)
@app.callback(Output('table', 'data'),
              Input('btn-nclicks-2', 'n_clicks')
)
def displayClick(btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-2' in changed_id:
        for entry in dbx.files_list_folder('').entries:
            aa=entry.name
            if aa=='Chest X-ray.csv':
                bb=entry.id
                resultresult =dbx.files_get_temporary_link(bb)
                cc=resultresult.link
        df=pd.read_csv(cc)
        data=df.to_dict('records')
        return data
@app.callback(Output('plot', 'figure'),
              Input('btn-nclicks-2', 'n_clicks')
)
def updates_charts(btn2):
    for entry in dbx.files_list_folder('').entries:
            aa=entry.name
            if aa=='Chest X-ray.csv':
                bb=entry.id
                resultresult =dbx.files_get_temporary_link(bb)
                cc=resultresult.link
    df=pd.read_csv(cc)
    y1=df['TB'].to_list()
    y2=df['Covid'].to_list()
    y3=df['PE'].to_list()
    x1=['TB','Covid','PE']
    y4=[*y1,*y2,*y3]
    fig = go.Figure([go.Bar(x=x1, y=y4,text=y4,marker=dict(color= "rgb(255, 127, 14)"),textposition='auto')])
    fig.layout.plot_bgcolor = '#0A061C'
    fig.layout.paper_bgcolor = '#0A061C '
    fig.update_geos(
    projection_type="orthographic",
    landcolor="white",
    oceancolor="MidnightBlue",
    showocean=True,
    lakecolor="LightBlue"
    )
    fig.update_layout(
    template="plotly_dark",
    margin=dict(r=10, t=25, b=40, l=60),
    annotations=[
        dict(
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=0)
    ]
    )
    fig.update_layout(title_text="Lyfas Rajorpay Dashboard")
    return fig
@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'According to doctor have Seletced result is "{}"'.format(value)
@app.callback(Output('container-button', 'children'),
              Input('btn-nclicks-3', 'n_clicks'))
def displayClick(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-3' in changed_id:
        pdf = FPDF('P', 'mm', 'Letter')
        pdf.add_page()
        pdf.set_font('times', 'B', 20)
        pdf.set_text_color(0,0,0)
        pdf.cell(0,10, 'Lyfas Analytics Report', 0, 1,'C')
        pdf.image('Mobile_02-PNG.png', x = 10, y = 15, w = 200, h = 70, type = '', link = 'www.lyfas.com')
        pdf.ln(60)
        pdf.set_font('times', '', 12)
        pdf.multi_cell(0,6,"Acculi Labs Pvt. Ltd. is a Point-of-care Preventive, Predictive &Personalized Digital Healthcare. Our tool Lyfas helps us to correlate various life events with the cognitive & behavioral psycho-dynamics of an individual. This is further mapped with the physiological symptoms and pathological observations.",border=True,align='J')
        #print(pdf.output)
        #pdf.image('Lyfas-logo-TM.png', x = 80, y = 10, w = 50, h = 20, type = '', link = 'www.lyfas.com')
        dbx = dropbox.Dropbox('5uSdWA0gd2UAAAAAAAAAAauPVaO_t_nlwRgP3YzwZ8-2HlxYFWRLUrmTAgk4F4b7')
        dbx.users_get_current_account()
        #dbx.files_delete_v2('/ter.pdf')
        dbx.files_upload(
                pdf.output(dest='S').encode('latin-1'),'/Patinet test.pdf',mode=dropbox.files.WriteMode.overwrite)
        with open("Pateint.pdf", "wb") as f:
            metadata, res = dbx.files_download(path="/Patinet test.pdf")
            f.write(res.content)
if __name__ == '__main__':
    app.run_server(debug=False)
