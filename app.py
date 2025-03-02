import streamlit as st
import joblib
test_model = joblib.load("lr_model.jb")


vectorizer = joblib.load("vectorizer.jb")
#joblib.dump(vectorizer, "vectorizer_compressed.jb", compress=3)  # You can use compress=3 to 9

print(test_model.predict(vectorizer.transform(["Breaking news! AI is amazing."])))

st.title('Fake news Detector')
st.write('Enter a news Articles below to check it is fake or Real ')

news_input = st.text_area("News Articles :" ,"")

if st.button('Check News'):
    if news_input.strip():
        transfron_input=vectorizer.transform([news_input])
        prediction = test_model.predict(transfron_input)


        if prediction[0]==1:
            st.success("The News is Real")

        else:
            st.error("The news is fake")
    else:
        st.warning("Please enter your test to analyze") 

