import streamlit as st

from inference import predict


def main():
    st.title("Audio Clasification with BEATs")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        prediction = predict(uploaded_file)
        st.audio(uploaded_file, format="audio/wav")
        st.write(f"Prediction: {prediction}")


if __name__ == '__main__':
    main()
