

import streamlit as st 
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tempfile import NamedTemporaryFile


@st.cache(allow_output_mutation=True)
def get_model():
        model = load_model('Model/trafficmodel.hdf5')
        print('Model Loaded')
        return model 

        
def predict(image):
        loaded_model = get_model()
        image = load_img(image, target_size=(150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image,axis=0)
        image = image/255.0

        classes = loaded_model.predict_classes(image)

        return classes



sign_names = {
        0: 'High Density Traffic',
        1: 'Low Density Traffic',
        2: 'Moderate Density Traffic'
       }

st.title("Traffic Density Analysis with Machine Learning")

st.write("A Project by: Hafiz Aiman Dinie")
st.write("Supervisor: Mdm Shahbe")

menu_select = st.sidebar.selectbox("Menu",("Prediction","Performance Comparison"))

st.set_option('deprecation.showfileUploaderEncoding', False)



                
                
def get_menu(menu_select):
    if menu_select == "Prediction":
        
        buffer = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        temp_file = NamedTemporaryFile(delete=False)
        if buffer:
            temp_file.write(buffer.getvalue())
            st.image(temp_file.name, caption='Uploaded Image', use_column_width=True)
        
            st.write("")

        if st.button('predict'):
                st.write("Result...")
                label = predict(temp_file.name)
                label = label.item()

                res = sign_names.get(label)
                st.title(res)
                
    elif menu_select == "Performance Comparison":
        st.title("") 
        st.title("Performance Comparison") 
        st.title("")
        st.write("1. Overall Model Accuracy Comparison")
        st.image("overallACC.png")
        st.title("")
        st.write("2. Comparison Model Accuracy by Traffic Density Category")
        st.image("accuracy2.png")
        
                
       

get_menu(menu_select)

