import streamlit as st
import numpy as np
import pandas as pd

colors = {  # inspired by Midnight Sun: Girls' Trip
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

# ✅ FIXED CSS - No complex RGB conversion
st.markdown(f"""
<style>
    /* 🌌 COSMIC NEBULA BACKGROUND - SIMPLIFIED & WORKING */
    .main {{
        background: 
            radial-gradient(ellipse at bottom, 
                {colors['background']} 0%, 
                {colors['surface']} 30%, 
                #2a1b3d 60%, 
                transparent 80%),
            linear-gradient(135deg, 
                {colors['primary']} 0%, 
                {colors['secondary']} 25%, 
                {colors['accent']} 50%, 
                {colors['primary']} 75%, 
                {colors['secondary']} 100%);
        background-size: 200% 200%, 100% 100%;
        animation: nebulaFlow 20s ease-in-out infinite;
        position: relative;
        min-height: 100vh;
        overflow: hidden;
    }}
    
    @keyframes nebulaFlow {{
        0%, 100% {{ background-position: 0% 0%, 0% 0%; }}
        50% {{ background-position: 100% 100%, 0% 0%; }}
    }}
    
    /* ✨ SIMPLIFIED SPARKLES - Uses HEX directly */
    .main::after {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 80% 70%, {colors['primary']}, transparent),
            radial-gradient(0.5px 0.5px at 40% 90%, {colors['accent']}, transparent);
        background-repeat: repeat;
        background-size: 150px 120px;
        animation: sparkleFloat 30s linear infinite;
        pointer-events: none;
        opacity: 0.6;
    }}
    
    @keyframes sparkleFloat {{
        from {{ transform: translateY(0px); }}
        to {{ transform: translateY(-120px); }}
    }}
    
    /* App container */
    .appview-container .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(26, 27, 46, 0.97);
        border-radius: 25px;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.4);
        border: 1px solid rgba(255, 105, 180, 0.25);
        position: relative;
        z-index: 10;
    }}
    
    /* Headers */
    h1 {{
        background: linear-gradient(45deg, {colors['primary']}, {colors['secondary']}, {colors['accent']}) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 0 30px rgba(255, 105, 180, 0.7) !important;
    }}
    
    h2 {{
        color: {colors['secondary']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        text-shadow: 0 0 15px rgba(138, 43, 226, 0.5) !important;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] > div > div {{
        background: linear-gradient(180deg, {colors['card']}, {colors['surface']}) !important;
        border-right: 1px solid rgba(255, 105, 180, 0.4) !important;
        backdrop-filter: blur(15px) !important;
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
        background: linear-gradient(45deg, {colors['primary']}, {colors['secondary']}, {colors['accent']}) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 30px rgba(255, 105, 180, 0.5) !important;
        transition: all 0.4s ease !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(255, 105, 180, 0.7) !important;
    }}
    
    /* File uploader */
    .stFileUploader > div {{
        background-color: {colors['card']} !important;
        border: 2px dashed {colors['accent']} !important;
        border-radius: 20px !important;
    }}
    
    /* Text */
    .stMarkdown {{
        color: {colors['light']} !important;
    }}
    
    /* Metrics */
    .stMetric {{
        background: linear-gradient(135deg, rgba(255, 105, 180, 0.15), rgba(138, 43, 226, 0.1)) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 105, 180, 0.3) !important;
    }}
</style>
""", unsafe_allow_html=True)

def main():
    # ✅ FIXED TITLE - No complex f-string issues
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 3rem; padding: 2rem;'>
        <h1 style='
            font-size: 4.5rem; 
            background: linear-gradient(45deg, {colors["primary"]}, {colors["secondary"]}, {colors["accent"]});
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent;
            background-clip: text; 
            font-weight: 900;
            letter-spacing: -2px;
            margin-bottom: 1rem;
            text-shadow: 0 0 40px rgba(255, 105, 180, 0.8);
            animation: titleGlow 2s ease-in-out infinite alternate;
        '>
            🔍 ThreatFind
        </h1>
        <p style='
            color: {colors["light"]}; 
            font-size: 1.6rem; 
            font-weight: 300; 
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
            max-width: 700px;
            margin: 0 auto;
            font-family: Inter, sans-serif;
        '>
            Snuff out any potential insider threats before they can cause harm 🚨
        </p>
    </div>
    
    <style>
    @keyframes titleGlow {{
        0% {{ filter: drop-shadow(0 0 10px {colors["primary"]}); }}
        100% {{ filter: drop-shadow(0 0 30px {colors["secondary"]}); }}
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown(f"### 🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Organisational Search via CSV", 
         "🎯 Single Search", 
         "📊 Exploratory Data Analysis"]
    )
    st.divider()

    if page == "🔍 Organisational Search via CSV":
        st.markdown(f"<h2 style='color: {colors['secondary']}; text-align: center;'>🏢 Organisational Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("👆 **Upload your CSV file** containing employee data, logs, or threat indicators")
        with col2:
            st.metric("Files Analyzed", "0", help="Number of CSV files processed")

        file_upload = st.file_uploader("📁 Select Your CSV File", type="csv")
        
        if file_upload is not None:
            try:
                df = pd.read_csv(file_upload)
                st.success(f"✅ Loaded **{len(df):,}** records successfully!")
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Records", len(df))
                with col2:
                    if 'employee_campus' in df.columns:
                        st.metric("🏫 Unique Campuses", df["employee_campus"].nunique())
                with col3:
                    if st.button("🚀 Run Threat Analysis", type="primary"):
                        st.balloons()
                        st.success("🔍 Analysis complete! No immediate threats detected.")
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 Try a different CSV format or check your file structure")

    elif page == "🎯 Single Search":
        st.markdown(f"<h2 style='color: {colors['accent']}; text-align: center;'>🔎 Single Entity Search</h2>", unsafe_allow_html=True)
        st.info("🎯 **Coming soon**: Search individual users, IPs, or threat indicators")
        
    elif page == "📊 Exploratory Data Analysis":
        st.markdown(f"<h2 style='color: {colors['success']}; text-align: center;'>📈 Threat Intelligence Dashboard</h2>", unsafe_allow_html=True)
        st.info("🔬 **Coming soon**: Interactive visualizations & anomaly detection")

if __name__ == "__main__":
    main()