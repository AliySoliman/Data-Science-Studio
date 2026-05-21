import streamlit as st
import time
import requests
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

from streamlit_option_menu import option_menu  # Requires pip install streamlit-option-menu
import  transform   ,machine  ,save , reporting,visualizations,stats, ai_recommendations_lib

from user_config import load_config
from setup_lib import boot_sequence

# Trigger Advanced Boot Sequence
if not boot_sequence():
    st.stop()

# If we get here, setup is complete and Ollama is ready
config = load_config()

st.markdown("""
<style>
[data-testid="stAppViewContainer"]  {
            background-image: url("https://github.com/lWAHBAl/img/raw/20c1f928721c741288562640559fc6e039d2b034/image.png");
            background-size: cover;
            background-attachment: local;
            background-position: center;
            background-repeat: no-repeat;
            }  

/* Broad, robust selectors that match Streamlit's DOM variants */
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stHeading"] h1,
.stApp .stHeading h1,
.stApp .stMarkdownContainer h1 {
  color: #1E90FF !important;   /* change to your color */
  font-family: "Arial", sans-serif !important;
  font-weight: 700 !important;
}
[data-testid= "stHeader"]{
            background-color: rgba(0,0,0,0.0);
            }

/* h2 / h3 examples */
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp .stHeading h2 {
  color: #28A745 !important;
}

.stApp [data-testid="stMarkdownContainer"] h3 {
  color: #FF1493 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Headings */
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stHeading"] h1,
.stApp .stHeading h1,
.stApp .stMarkdownContainer h1 {
  color: #7B99C4 !important;
  font-family: "Arial", sans-serif !important;
  font-weight: 700 !important;
}

/* h2 / h3 */
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp .stHeading h2 {
  color: #FFFFFF !important;
}
.stApp [data-testid="stMarkdownContainer"] h3 {
  color: #FF1493 !important;
}
.stApp [data-testid="stExpander"]  {
  border:2px solid;
  color : white !important;
  border-radius: 20px !important; 
  background-color: #0e2440 !important;
}

/* Paragraphs */
.stApp p {
    color: #FFFFFF !important;
    font-size: 20px !important;
    line-height: 1.6 !important;
}
/* Secondary buttons */
div[data-testid="stButton"] > button[kind="secondary"] {
    background-color: #161a1d !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5em 1em !important;
    font-weight: 600 !important;
}

/* Navigation bar tweaks */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
div[data-testid="stHorizontalBlock"] {
    margin-bottom: 2rem;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 0.5rem;
}

/* Background image */

</style>
""", unsafe_allow_html=True)
        # .stApp {
        # background-color: #091030; /* Light grey */
        #     }
# if "workflow" not in st.session_state :
#       st.session_state.workflow = False
# if st.button("workflow mode"):
#      st.session_state.workflow = True
col1,col2 = st.columns([5,1])
# Create top navigation bar
# if not st.session_state.workflow :
with col1:
    selected = option_menu(
        menu_title=None,
        options=["AI Recommendations", "Transform", "Analysis and EDA","Machine Learning","stats","Reporting"],
        icons=["speedometer2", "gear", "speedometer2"],  # Bootstrap icons
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {  
                "padding": "0!important", 
                "background-color": "#7B99C4",
                "margin-bottom": "1rem"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "padding": "8px 12px",
                "border-radius": "4px"
            },
            "nav-link-selected": {
                "background-color": "#0F1533",
                "color": "white"
            }
        }
    )

with col2:

# Apply class to buttons

        if st.button("Save Progress",type="secondary",use_container_width=True):
            #st.write(st.session_state.pipeline)
            save.save_encoded_transformations(transformations_dict=st.session_state.pipeline)
            try:
                save.save_ai_recommendations(ai_recommendations=st.session_state.ai_recommendations)
            except AttributeError:
                pass
            # current_pipeline = st.session_state.pipeline
        # if selected == "Transform":
        #     transform.run()
        # elif selected == "Analysis and EDA":
        #     visualizations.run()
        # elif selected == "Machine Learning":
        #     machine.run()
        # elif selected == "Reporting":
        #     reporting.run()
        # elif selected == "stats":
        #     stats.run()
if selected == "AI Recommendations":
    ai_recommendations_lib.run()
if selected == "Transform":
    transform.run()
elif selected == "Analysis and EDA":
    visualizations.run()
elif selected == "Machine Learning":
    machine.run()
elif selected == "Reporting":
    reporting.run()
elif selected == "stats":
    stats.run()
# elif selected == "workflow": 
#      workflow.run()
