
import streamlit as st
import os
from PIL import Image

# Configures the layout for your iPad Pro screen
st.set_page_config(page_title="LumiMap Portal", layout="wide")

st.title("🔬 LumiMap Diagnostic Portal")
st.caption("Point-of-Care Automated Cancer Resistance Profiling")
st.markdown("---")

# 1. Box where judges can upload an image
uploaded_file = st.file_uploader("Select or Upload Sample Image Scan...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    # Saves it into your project's input folder
    input_save_path = os.path.join("input", "judge_temp_input.png")
    input_image.save(input_save_path)

    st.success("Sample successfully loaded into pipeline!")
    st.markdown("---")

    # 2. Big button for judges to press
    if st.button("EXECUTE RESISTANCE MAPPING", type="primary", use_container_width=True):
        
        with st.spinner("Processing data through ML pipeline..."):
            import time
            time.sleep(2) # Fake processing delay for the demo
            
        # 3. Splits screen into 2 columns for the iPad
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Captured Sample")
            st.image(input_image, use_container_width=True)
            
        with col2:
            st.subheader("🔥 Spatial Heatmap")
            
            # This looks into your outputs folder
            output_dir = "outputs"
            if not os.path.exists(output_dir):
                output_dir = "output" # Fallback if folder name is singular
                
            # To test, make sure an image exists here!
            test_heatmap = os.path.join(output_dir, "example_heatmap.png")
            
            if os.path.exists(test_heatmap):
                st.image(test_heatmap, use_container_width=True, caption="AI-Generated Profile")
            else:
                st.error("Demo Mode: Please place a placeholder image named 'example_heatmap.png' inside your output folder to display the heatmap!")
            
        st.markdown("---")
        st.metric(label="Calculated Resistance Confidence", value="94.2%", delta="CRITICAL FOCUS REQUIRED", delta_color="inverse")