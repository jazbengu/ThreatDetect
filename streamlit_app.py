import streamlit as st
import numpy as np
import pandas as pd

colors = { # inspired by Mightnight Sun: Girls' Trip
    'primary': '#FF69B4',      # Hot pink (main brand color)
    'secondary': '#8A2BE2',    # Blue violet
    'accent': '#00CED1',       # Dark turquoise  
    'background': '#0F0F23',   # Deep midnight blue
    'surface': '#1A1B2E',      # Dark purple-gray
    'card': '#16213E',         # Navy blue
    'success': '#FF1493',      # Deep pink
    'warning': '#FF4500',      # Orange red
    'light': '#E6E6FA',        # Lavender blush
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

st.markdown(f"""
<style>
    /* Main background gradient */
    .main {{
        background: linear-gradient(135deg, 
            {colors['background']} 0%, 
            {colors['surface']} 50%, 
            {colors['gradient_start']} 100%);
        background-attachment: fixed;
    }}
    
    /* App container */
    .appview-container .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(26, 27, 46, 0.95);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 105, 180, 0.2);
    }}
    
    /* Headers */
    h1 {{
        color: {colors['primary']} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(255, 105, 180, 0.5);
        font-size: 3.5rem !important;
        margin-bottom: 1rem !important;
    }}
    
    h2 {{
        color: {colors['secondary']} !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.2rem !important;
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background: linear-gradient(180deg, 
            {colors['card']} 0%, 
            {colors['surface']} 100%);
        border-right: 1px solid rgba(255, 105, 180, 0.3);
        backdrop-filter: blur(10px);
    }}
    
    /* Sidebar selectbox */
    .stSelectbox > div > div {{
        background-color: {colors['card']} !important;
        border: 1px solid {colors['accent']} !important;
        border-radius: 12px !important;
        color: {colors['light']} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(45deg, 
            {colors['primary']}, 
            {colors['secondary']});
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 25px rgba(255, 105, 180, 0.4) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(255, 105, 180, 0.6) !important;
    }}
    
    /* File uploader */
    .stFileUploader > div {{
        background-color: {colors['card']} !important;
        border: 2px dashed {colors['accent']} !important;
        border-radius: 15px !important;
    }}
    
    /* Text */
    .stMarkdown {{
        color: {colors['light']} !important;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Metrics and dataframes */
    .stMetric {{
        background: rgba(255, 105, 180, 0.1) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 105, 180, 0.3) !important;
    }}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 4rem; background: linear-gradient(45deg, #FF69B4, #8A2BE2, #00CED1); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text; animation: glow 2s ease-in-out infinite alternate;'>
                🔍 ThreatFind
            </h1>
            <p style='color: #E6E6FA; font-size: 1.4rem; font-weight: 300; 
                    text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);'>
                Snuff out any potential insider threats before they can cause harm.
            </p>
        </div>
        """, unsafe_allow_html=True)

    page = st.sidebar.selectbox(
    "Choose a page:",
    ["Organisational Search via CSV", "Single Search", "Exploratory Data Analaysis"]
)
    st.divider()

    if page == "Organisational Search via CSV":
        st.markdown(f"<h2 style='color: {colors['secondary']};'>Organisational Analysis</h2>", unsafe_allow_html=True)
        
        column_1, column_2 = st.columns([2,1])
        with column_1:
            st.info("👆 **Upload your CSV file** containing employee data, logs, or threat indicators")
        with column_2:
            st.metric("Files Analyzed", "0", help="Number of CSV files processed")

        file_upload = st.file_uploader("Select Your File", type="csv")
        if file_upload is not None:
            try:
                df = pd.read_csv(file_upload)
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
                st.metric("Files Analyzed", "1")
                col_1,col_2,col_3 = st.columns(3)
                with col_1:
                    st.metric("Total Records", f"{len(df)}")
                    st.metric("Unique Campuses", df["employee_campus"].nunique())
                with col_3:
                    st.button("🚀 Run Threat Analysis", type="primary")
            except Exception as e:
                st.error(f"Error processing file: {e}")


    elif page == "Single Search":
        st.title("Single Search")
        # Chart code here

    elif page == "Exploratory Data Analaysis":
        st.title("EDA")
        # Settings code here

   
    
if __name__=="__main__":
    main()


   

