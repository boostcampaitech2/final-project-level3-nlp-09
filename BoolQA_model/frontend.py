
import requests
import streamlit as st
import time
import json
from random import *

def main():
    st.title('YesorNo Classification')
    text_input= st.text_input(f"질문을 입력해주세요. (exit 입력시 종료)")

    if 'text_list' not in st.session_state or 'predict_list' not in st.session_state:
        st.session_state.idx= randint(0, 10)
        st.session_state.text_list= []
        st.session_state.predict_list= []
        
    if st.button('문제 다시 생성'):
        st.session_state.text_list= []
        st.session_state.predict_list= []
        st.session_state.idx= randint(0, 10)


    if text_input:
        # print(text_input)
        # print(st.session_state.idx)
        text= json.dumps({'text': text_input, 'passage_idx': st.session_state.idx})
        # text= json.dump({'text': text_input})
        
        response= requests.post('http://0.0.0.0:5000/predict', data= text)
        label= response.json()  

        st.session_state.text_list.append(text_input)
        # st.session_state.predict_list.append(label)
        if label['label']== 'exit':
            st.write(f"label : {label['answer']}")
            st.text_area(label['passage'])

        elif label['label'] == 1:
            st.session_state.predict_list.append('네. 맞습니다.')
        elif label['label'] == 0:
            st.session_state.predict_list.append('아닙니다.')    
        elif label['label'] == 2:
            st.session_state.predict_list.append('잘 모르겠습니다.')  

        

        for i, j in zip(st.session_state.text_list, st.session_state.predict_list):
            st.write('user :', i)
            st.text(f".\t\t\t\t\t\t\t\tchatbot : {j}")

    
if __name__ == '__main__':
    main()