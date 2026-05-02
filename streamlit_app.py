import streamlit as st

def main():
    st.title("ThreatFind")
    st.write("snuff out any potential insider thereats \n\n\n\n")

    page = st.sidebar.selectbox(
    "Choose a page:",
    ["Organisational Search via CSV", "Single Search", "Exploratory Data Analaysis"]
)

    if page == "Organisational Search via CSV":
        st.title("Dashboard")
        # add line for enter csv file and prompt
        st.write("Please Enter Your CSV File For Analysis")
        file_upload = st.file_uploader("Select Your File", type="csv")


    elif page == "Single Search":
        st.title("Single Search")
        # Chart code here

    elif page == "Exploratory Data Analaysis":
        st.title("EDA")
        # Settings code here

   
    
if __name__=="__main__":
    main()

