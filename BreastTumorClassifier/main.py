import streamlit as st
import os
from PIL import Image
import io
from roboflow import Roboflow
from utils import is_valid_image, preprocess_image
import time

# Configure page with a wider layout
st.set_page_config(
    page_title="Medical Image Analysis | Tumor Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add logo
logo_col1, logo_col2, logo_col3 = st.columns([1, 3, 7])
with logo_col1:
    st.image("saved-image.png", width=750)


def get_roboflow_model():
    """Initialize Roboflow and get the model with proper error handling"""
    try:
        rf = Roboflow(api_key="9nNCAU4ruTqoGLrVfMn9")
        workspace = rf.workspace("hatch-cell-struture")

        try:
            # Get the specific project
            project = workspace.project("my-first-project-epj8p-sjzvo")
            # Get version 2 of the model
            model = project.version(2).model
            return model

        except Exception as e:
            st.error(f"Error accessing project: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Error connecting to Roboflow: {str(e)}")
        st.info(
            "Please ensure your Roboflow API key is correct and you have proper access permissions."
        )
        return None


def main():
    # Header section with improved layout
    st.markdown("""
        <h1 style='text-align: center;'>
            Advanced Tumor Classification System
        </h1>
        <p style='text-align: center; font-size: 1.2rem; color: #4a5568; margin-bottom: 2rem;'>
            Powered by AI for Accurate Medical Image Analysis
        </p>
    """, unsafe_allow_html=True)

    # Initialize Roboflow model
    model = get_roboflow_model()
    if not model:
        st.stop()

    # Main content with improved layout and centered columns
    st.markdown("<div class='centered-content'>", unsafe_allow_html=True)

    # Centered upload section
    st.markdown("<h2>Image Upload</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 20px;'>Choose a mammogram image</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        " ",  # Empty label since we're using custom text above
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a clear, high-quality medical image for best results"
    )

    if uploaded_file:
        if not is_valid_image(uploaded_file):
            st.error("Please upload a valid medical image file")
            return

        # Create columns for centered image display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image", key="analyze"):
            with st.spinner("Processing image..."):
                try:
                    # Process image and get prediction
                    processed_image = preprocess_image(image)
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

                            # Enhanced result box
                            result_color = "#29b910" if class_name.lower() == "benign" else "#EF4444"
                            st.markdown(f"""
                                <div class="result-box" style="border-color: {result_color}">
                                    <h3 style="text-align: center; margin-bottom: 1rem;">Classification Result</h3>
                                    <p class="result-text" style="color: {result_color}">
                                        {class_name.upper()}
                                    </p>
                                    <div class="confidence">
                                        Confidence Level: {confidence:.1f}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            # Detailed interpretation
                            st.markdown("### Detailed Analysis")
                            if class_name.lower() == "benign":
                                st.success("""
                                    The analysis indicates a **benign tumor pattern**.

                                    **Recommendations:**
                                    - Schedule a follow-up with your healthcare provider
                                    - Continue regular check-ups
                                    - Maintain your medical records
                                """)
                            else:
                                st.error("""
                                    The analysis indicates a **malignant tumor pattern**.

                                    **Immediate Actions Recommended:**
                                    - Consult with your healthcare provider as soon as possible
                                    - Bring this analysis to your appointment
                                    - Request comprehensive evaluation
                                """)
                        else:
                            st.warning(
                                "Analysis inconclusive. Please ensure the image is clear and "
                                "try again, or use a different image."
                            )

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Medical Disclaimer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <h4>Medical Disclaimer</h4>
            <p>This tool is designed to assist medical professionals and should not be used as a substitute for professional medical diagnosis.
            <p style='font-size: 0.8rem; margin-top: 1rem;'>© 2025 Theravos Medical Group</p>
        </div>
    """, unsafe_allow_html=True)

    # Information section (moved from sidebar)
    st.markdown("## About")
    st.info(
        "This advanced medical imaging system uses AI to analyze and classify "
        "breast tumors. Upload a medical image to receive instant analysis."
    )

    st.markdown("### Supported Formats")
    st.markdown("""
        - JPEG/JPG
        - PNG
        - BMP

        ### Analysis Process
        1. Upload your image
        2. Click 'Analyze Image'
        3. Review the results

    """)


if __name__ == "__main__":
    main()
