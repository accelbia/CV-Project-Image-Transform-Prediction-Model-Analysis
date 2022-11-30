import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
st.set_page_config(layout="wide")

st.title('Image Transform Prediction Model Analysis - About')
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


col1, col2 = st.columns(2)

with col1:
    st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_fclga8fl.json"), quality = 'high')

with col2:
    st.header('What we tried to do, was to find the best extractor for specific scenarios.')
    st.text('A Wavelet is a wave-like oscillation that is localized in time, an example \nis given below. Wavelets have two basic properties: scale and location. \nScale (or dilation) defines how “stretched” or “squished” a wavelet is. \nThis property is related to frequency as defined for waves. \nLocation defines where the wavelet is positioned in time (or space).')
    st.subheader('Creators')
    
    st.markdown('CB.EN.U4CSE19106 - [Ayush Barik](http://github.com/accelbia)')
    st.markdown('CB.EN.U4CSE19106 - [T. Ganesh Rohit Sarma](http://github.com/roathena)')
    st.markdown('CB.EN.U4CSE19106 - [A Shivaani](http://github.com/ShivaaniAnand)')
    st.markdown('CB.EN.U4CSE19106 - [V Neelesh Gupta](http://github.com/vayigandlaneelesh)')
    # ref = {
    #     'Roll no.' : ['CB.EN.U4CSE19106','CB.EN.U4CSE19106','CB.EN.U4CSE19106','CB.EN.U4CSE19106'],
    #     'Name' : ['T. Ganesh Rohit Sarma','Ayush Barik','A Shivaani','V Neelesh Gupta']
    # }
    # st.write(pd.DataFrame(ref))