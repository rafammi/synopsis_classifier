import streamlit as st
from src import model

THRESHOLD = 0.5

@st.cache_resource
def load_predictor():
    predictor = model.genrePredictor(num_classes=0, class_names=[])
    predictor.load_model()
    return predictor

st.title("Movie Genre Classifier")
st.write("Enter a movie synopsis and the model will predict its genres.")

overview = st.text_area("Synopsis", height=150)

if st.button("Predict") and overview.strip():
    predictor = load_predictor()
    probs = predictor.predict(overview)

    thresholds = predictor.thresholds if predictor.thresholds is not None else THRESHOLD
    predicted = probs[probs >= thresholds]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted genres")
        if predicted.empty:
            st.info("No genre exceeded the threshold.")
            st.dataframe(probs.head(5).rename("probability").to_frame().style.format("{:.1%}"))
        else:
            st.dataframe(predicted.rename("probability").to_frame().style.format("{:.1%}"))

    with col2:
        st.subheader("All probabilities")
        st.bar_chart(probs)
