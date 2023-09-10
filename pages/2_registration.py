import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from home import face_recg

registration_form=face_recg.RegistrationForm()

st.subheader('Registration_Form')

person_name=st.text_input(label='Name',placeholder='First and Last Name')
role=st.selectbox(label='select Role',options=('student','Teacher'))

if st.button('Submit'):
    st.write(f'person_name=',person_name)
    st.write(f'role=',role)


def video_call_back(frame):
    img=frame.to_ndarray(format='bgr24')
    ret,embed=registration_form.get_embeddings(img)
    if embed is not None:
        with open('face_embed.txt',mode='ab') as f:
            np.savetxt(f,embed)
            

    return av.VideoFrame.from_ndarray(ret,format='bgr24')
    
webrtc_streamer(key='registration',video_frame_callback=video_call_back)


if st.button('submit'):
    return_val=registration_form.save_data_redis(person_name,role)
    if(return_val==True):
        st.success("{person_name} registered succesfully")
    elif return_val=='name_false':
        st.error('name cant be empty,name cannot be empty ot spaces')

    elif return_val=='file_fase':
        st.error('face_embed.txt not found,refresh')
