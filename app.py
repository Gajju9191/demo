import numpy as np
import pandas as pd
import cv2 as cv2
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit as st

@st.cache(allow_output_mutation = True)
def load_mod():
    model = load_model("data/0.2171-0.9380.h5")
    return model

@st.cache(allow_output_mutation = True)
def load_csv():
    birds_df = pd.read_csv("data/birds_species.csv")
    return birds_df

model = load_mod()
birds_df = load_csv()
test_df = birds_df[birds_df['Dataset'] == 'test'].reset_index()
classes = birds_df['Species'].unique()
height = 128
width =128

st.markdown("# Welcome To Bird Species Identification!:bird:")
st.markdown("""Upload your image of a bird and the Species of the bird
            will be will be predicted by a Deep Neural Network in
            real-time and displayed on the screen. Top Five most
            likely Bird Species along with confidence level will
            also be displayed.
            """)
st.info("Data obtained from the [Kaggle Dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) by Gerald Piosenka.")

file = st.file_uploader("Upload A Bird Image")
if file:
    image = Image.open(file)
    image_path = r"data\image.jpg"
    image.save(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (height, width), interpolation = cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis = 0)
    y_pred_prob = model.predict(image, verbose = 0)
    y_pred = np.argmax(y_pred_prob, axis = 1)[0]
    y_top5_prob = np.sort(y_pred_prob)[:, -1:-6:-1]
    y_top5_label = np.argsort(y_pred_prob)[:, -1:-6:-1]
    top5 = list(zip(y_top5_label[0], y_top5_prob[0]))
    
    st.markdown("## Here is the Image You have uploaded.")
    st.image(image_path)
    st.success(f"The Bird belongs to **{classes[y_pred]}** Species.")
    
    df = pd.DataFrame(data = np.zeros((5, 2)), columns = ['Species', 'Confidence Level'],
              index = np.linspace(1, 5, 5, dtype = int), dtype = 'string')
    for i, (label, prob) in enumerate(top5):
        df.iloc[i, 0] = classes[label]
        df.iloc[i, 1] = str(np.round((prob * 100), 4)) + "%"
    st.markdown("## Here are the five most likely Bird Species.")
    st.dataframe(df)
    st.markdown(f"## Here are some other images of {classes[y_pred]}.")
    lst = []
    for i in range(3):
        path = test_df[test_df['Species'] == classes[y_pred]]['Filepath'].values[i]
        lst.append(path)
    st.image(lst)
