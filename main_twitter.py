
import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from PIL import Image

pickle_in = open('model.pkl',"rb")
classifier=pickle.load(pickle_in)

def predict_turnover(in_data):
    df = pd.DataFrame(in_data)
    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=8, stop_words='english')
    # bag-of-words feature matrix
    x = vectorizer.fit_transform(df[0])
    prediction=classifier.predict_proba(x)
    print(prediction)
    predictionF = prediction[:,1] >= 0.45 # if prediction is greater than or equal to 0.45 than 1 else 0

    
    if predictionF == True:
        return "The Comment is Negative"
    return "The Comment is Positive"



def main():
  st.title("Sentiment Analysis - Twitter Comments")
  
  html_temp = """
  <div style="background-color:tomato;padding:10px">
  <h2 style="color:white;text-align:center;"> Predict Employee Turnover Web APP </h2>
  </div>
  """
  list=[]
  st.markdown(html_temp,unsafe_allow_html=True)
  if st.button("About this ML Project"):
      st.markdown('Employee turnover prediction web app is machine learning based classification model' 
        ' which helps to determine turnover of employees for the organization. '
        ' It will help companies to know about employee turnover in advance to '
        ' provide them sufficient time to take necessary decisions.')
    
  input1 = st.text_input('Movie title')
  #title = st.text_input('Movie title', 'Life of Brian')
  input1 = str(input1)

   
    
  inputs = []
  inputs.append(input1)
  inputs = np.array(inputs)
  print(inputs)
  
  
  
  
  result=""
  if st.button("Predict"):
      result=predict_turnover(inputs)
  st.success('The output is {}'.format(result))
  if st.button("About"):
      st.markdown("Connect wtih me: [LINK](https://www.linkedin.com/in/shubham-deshmukh-b8a7691b0/)")
        #st.text("Connect With Me: https://www.linkedin.com/in/shubham-deshmukh-b8a7691b0/")
        


if __name__=='__main__':
    main()