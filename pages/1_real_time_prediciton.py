from home import st
from home import face_recg
st.set_page_config(page_title='predictions',layout='wide')
st.subheader('real time prediction')
import pandas as pd
import numpy as np
import redis
import av
import time
from streamlit_webrtc import webrtc_streamer
with st.spinner('Retriving database...'):
    
        redis_face_db=face_recg.retrive(name='academy:reister2')
        st.dataframe(redis_face_db)
st.success('Data succesfully retrived from redis')
waittime=30
settime=time.time()
realtimepred=face_recg.RealTimePred()

def video_frame_callback(frame):
        global settime
        img=frame.to_ndarray(format="bgr24")
        pred_img=realtimepred.prediction(img,redis_face_db,'features',['Name','Role','features'],cosine_optimal=0.5)
        timenow=time.time()
        difftime=timenow-settime
        if difftime>=waittime:
             realtimepred.saveLogs_redis()
             settime=time.time()
             

        return av.VideoFrame.from_ndarray(pred_img,format="bgr24")

webrtc_streamer(key="pred",video_frame_callback=video_frame_callback)

