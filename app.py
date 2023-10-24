import streamlit as st

from middleware import upload_to_s3, inference

BUCKET = 's3-sqs-lambda-ecs'


def main():
    st.title("Audio Clasification with BEATs")
    audio_file = st.file_uploader("Upload an audio file", type=["wav"])
    label, code = inference()

    if audio_file is not None:
        upload_to_s3(audio_file, BUCKET)
        st.audio(audio_file, format="audio/wav")
        st.write(f"Prediction: {label} (code:{code})")


if __name__ == '__main__':
    main()
