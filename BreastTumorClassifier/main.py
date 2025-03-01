import streamlit as st
import os
from PIL import Image
import io
from roboflow import Roboflow
from utils import is_valid_image, preprocess_image
import time

# Configure page
st.set_page_config(
    page_title="Breast Tumor Classification",
    page_icon="🏥",
    layout="wide"
)

# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_roboflow_model():
    """Initialize Roboflow and get the model with proper error handling"""
    try:
        rf = Roboflow(api_key="9nNCAU4ruTqoGLrVfMn9")
        workspace = rf.workspace("hatch-cell-struture")

        # Debug: Display workspace information
        st.sidebar.info("Attempting to connect to Roboflow workspace...")

        try:
            # Get the specific project
            project = workspace.project("my-first-project-epj8p-sjzvo")
            st.sidebar.success("Connected to project successfully!")

            # Get version 2 of the model
            model = project.version(2).model
            st.sidebar.success("Model loaded successfully!")
            return model

        except Exception as e:
            st.error(f"Error accessing project: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Error connecting to Roboflow: {str(e)}")
        st.info("Please ensure your Roboflow API key is correct and you have proper access permissions.")
        return None

def main():
    st.title("Breast Tumor Classification System")
    st.markdown("### AI-Powered Medical Image Analysis")

    # Initialize Roboflow model
    model = get_roboflow_model()
    if not model:
        st.stop()

    # Sidebar information
    with st.sidebar:
        st.markdown("## About")
        st.info(
            "This application uses advanced AI to classify breast tumors as either "
            "benign or malignant. Upload a medical image to get started."
        )
        st.markdown("### Supported Formats")
        st.write("- JPEG/JPG\n- PNG\n- BMP")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("## Image Upload")
        uploaded_file = st.file_uploader(
            "Upload a breast tumor image",
            type=['png', 'jpg', 'jpeg', 'bmp']
        )

        if uploaded_file:
            if not is_valid_image(uploaded_file):
                st.error("Please upload a valid image file")
                return

            image = Image.open(uploaded_file)
            # Updated to use use_container_width instead of use_column_width
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Analyze Image", key="analyze"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Process image and get prediction
                        processed_image = preprocess_image(image)
                        # Save temporary file for Roboflow
                        temp_path = "temp_image.jpg"
                        processed_image.save(temp_path)

                        # Get prediction
                        prediction = model.predict(temp_path, confidence=40, overlap=30).json()
                        os.remove(temp_path)  # Clean up

                        # Display results in the second column
                        with col2:
                            st.markdown("## Analysis Results")

                            if prediction['predictions']:
                                pred = prediction['predictions'][0]
                                confidence = pred['confidence'] * 100
                                class_name = pred['class']

                                # Result box
                                result_color = "#28a745" if class_name == "benign" else "#dc3545"
                                st.markdown(
                                    f"""
                                    <div class="result-box" style="border-color: {result_color}">
                                        <h3>Classification Result</h3>
                                        <p class="result-text" style="color: {result_color}">
                                            {class_name.upper()}
                                        </p>
                                        <div class="confidence">
                                            Confidence: {confidence:.1f}%
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                                # Additional information
                                st.markdown("### Interpretation")
                                if class_name == "benign":
                                    st.success(
                                        "The analysis suggests a benign tumor. However, please "
                                        "consult with a healthcare professional for proper diagnosis."
                                    )
                                else:
                                    st.error(
                                        "The analysis suggests a malignant tumor. Immediate "
                                        "consultation with a healthcare professional is recommended."
                                    )
                            else:
                                st.warning("No clear classification could be made. Please try with a different image.")

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p> This tool should not be used as a substitute for professional medical diagnosis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
