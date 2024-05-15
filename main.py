import datetime
from io import BytesIO
import os
import schedule
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import gspread
import gspread_dataframe
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import ObjectProperty
import gspread
from google.oauth2.service_account import Credentials

class CameraPreview(Image):
    def __init__(self, **kwargs):
        super(CameraPreview, self).__init__(**kwargs)
        # 0番目のカメラに接続
        self.capture = cv2.VideoCapture(0)
        # 描画のインターバルを設定
        Clock.schedule_interval(self.update, 1.0 / 30)

    # インターバルで実行する描画メソッド
    def update(self, dt):
        # フレームを読み込み
        ret, self.frame = self.capture.read()
        # Kivy Textureに変換
        buf = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # インスタンスのtextureを変更
        self.texture = texture
        schedule.run_pending()

        
class FB_Manager():
    def __init__(self, preview:CameraPreview, message:Label) -> None:
        self.face_detector = cv2.FaceDetectorYN.create("./model/yunet.onnx", "", (0, 0))
        self.face_recognizer = cv2.FaceRecognizerSF.create("./model/face_recognizer_fast.onnx", "")
        self.preview:CameraPreview = preview
        self.message:Label = message


        scope = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']
        self.credentials = Credentials.from_service_account_file("./secrets/credentials.json", scopes=scope)
        
        gc = gspread.authorize(self.credentials)

        SPREADSHEET_KEY = '1WjOhYLdUXSbuuzTZjhiM5uKm2SpMy4NSOE1JeE4y3Xk'
        SPREADSHEET_KEY_SECRET = '1eGA2QZ6QFQweBmWVTqtUT9d0RpU33x63d9F3uBsJQ6c'
        self.workbook = gc.open_by_key(SPREADSHEET_KEY)
        self.workbook_secret = gc.open_by_key(SPREADSHEET_KEY_SECRET)
        self.__makedict()

    def __makedict(self):
        service = build('drive', 'v3', credentials=self.credentials)

        drive = None

        folder_id = "1uGKHFEDPz47RSnW_3LjGJ1Ww06izOny_"

        results = service.files().list(q=f"'{folder_id}' in parents",
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name)').execute()
        items = results.get('files', [])

        self.table = df(self.workbook_secret.worksheet("table").get_all_records())
        self.feature_list = []
        if not items:
            print('No files found.')
        else:
            for file in items:
                print(u'{0} ({1})'.format(file['name'], file['id']))
                request = service.files().get_media(fileId=file['id'])
                fh = BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"Download {int(status.progress() * 100)}%.")
                
                # バイトデータをNumPy配列に変換
                fh.seek(0)
                image_bytes = np.asarray(bytearray(fh.read()), dtype=np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                self.feature_list.append((os.path.splitext(file['name'])[0],self.FaceFeature(image)))
                    
    def FaceFeature(self, frame:np.ndarray):
        height, width, _ = frame.shape
        self.face_detector.setInputSize((width, height))
        
        _, faces = self.face_detector.detect(frame)
        print(faces)
        if faces is None:
            return False
        return self.face_recognizer.feature(self.face_recognizer.alignCrop(frame, faces[0]))
    
    def match(self, feature):
        COSINE_THRESHOLD = 0.363
        maximum_score = COSINE_THRESHOLD
        recognized_person:pd.Series = None
        for value in self.feature_list:
            score = self.face_recognizer.match(feature,value[1], cv2.FaceRecognizerSF_FR_COSINE)
            print(f"{value[0]}-{score}")
            if score > maximum_score:
                recognized_person = value[0]
        return recognized_person if recognized_person is not None else False
                

class RecognizeButton(ButtonBehavior, Image):
    
    preview:CameraPreview = ObjectProperty(None)
    message:Label = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def init_fb_manager(self):
        self.fbm = FB_Manager(preview=  self.preview, message=self.message)
        self.reset()

    def on_press(self):
        frame = np.array(self.preview.frame)
        f = self.fbm.FaceFeature(frame)
        recognized_person = self.fbm.match(f)
        if recognized_person is not False:
            num = recognized_person
            name = self.fbm.table.loc[self.fbm.table["number"] == int(num),"name"].values[0]
            self.send_result(num,name)
            self.message.text = f"{name} 点呼済み"


        else:
            self.message.text = "誰も検知しませんでした"

    def send_result(self,num:int,name:str):
        sheet_names = [sheet.title for sheet in self.fbm.workbook.worksheets()]
        sheet_name = datetime.datetime.today().strftime('%Y-%m-%d')
        if sheet_name not in sheet_names:
            self.fbm.workbook.add_worksheet(title=sheet_name,rows=2,cols=1)
        worksheet = self.fbm.workbook.worksheet(sheet_name)
        worksheet.add_rows(1)
        row_count = worksheet.row_count
        worksheet.update_cell(row_count - 1,1,num)
        worksheet.update_cell(row_count - 1,2,name)

    def reset(self):
        pass




        

class MainScreen(Widget):
    pass

class DormAttApp(App):
    def build(self):
        MS = MainScreen()
        return MS
    
    def on_start(self):
        self.root.ids.recognize_button.init_fb_manager()


if __name__ == "__main__":
    DormAttApp().run()