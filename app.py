import streamlit as st
import os
from dotenv import load_dotenv
from textwrap import dedent
from phi.assistant import Assistant
#from phi.tools.newspaper_toolkit import NewspaperToolkit
from phi.tools.newspaper4k import Newspaper4k
#from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.openai import OpenAIChat
from st_copy_to_clipboard import st_copy_to_clipboard
import base64

from utils.assistants import Publisher, Planner, Writer, Editor
from utils.ui.agent_config_layout import get_config
from utils.ui.output_drafts import getOutput_drafts

load_dotenv()

st.set_page_config(page_title="AI Journalist", page_icon="🗞️", layout="wide")

from utils.page_config import getlogo, page_config

getlogo()
page_config()

# Dummy credentials
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Fixed API key (replace with your actual API key)
FIXED_API_KEY = os.getenv("OPENAI_API_KEY")


# Create a simple login function
def login(username, password):
    return username == USERNAME and password == PASSWORD

def main_app(api_key):
    with st.container():
        st.title("AI Journalist🗞️")
        st.caption("Generate high-quality articles with AI Journalist by researching, writing, and editing articles using GPT-4o.")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # Create Assistants
    planner = Planner()
    planner = planner.create_assistant()
    
    writer = Writer()
    writer = writer.create_assistant()
    
    editor = Editor()
    editor = editor.create_assistant()
    
    publisher = Publisher()
    publisher = publisher.create_assistant()
    #publisher.team = [planner, writer, editor]
    
    response = ""
    planner_response = ""
    writer_response = ""
    editor_response = ""
   
    tab1, tab2, tab3 = st.tabs(["\u2001Article\u2001\u2001", "\u2001\u2001Agent Config\u2001", "\u2001\u2001Agent Drafts\u2001"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.header("Input & Configuration")
                
                # Get user input
                query = st.text_input("What do you want the AI journalist to write an article on?", placeholder="E.g: Emergence of AI and LLMs.")
                
                # Get user preferences
                word_limit = st.slider("How long should be your article?", min_value=250, max_value=1500, step=50, key="word_limit")

                use_links = st.radio("Do you want to provide reference links?", ("No", "Yes"))

                links = []
                if use_links == "Yes":
                        num_links = st.number_input("How many links do you want to provide?", placeholder="Enter the number of links that you want to use", min_value=1, max_value=5, step=1, key="number_of_links", help="These links will be used to curate your news article.")
                        for i in range(num_links):
                            link = st.text_input(f"Link: {i+1}", key=f"link_{i+1}")
                            links.append(link)
            
                if use_links == "No" or (use_links == "Yes" and all(links)):
                    if st.button("Generate Article"):
                        if query:
                            with st.spinner("Good things take time, and we're making sure it's perfect for you!"):
                                
                                # Prepare the content for the writer
                                links_text = "\n".join(links) if links else "No reference links provided."
                                
                                planner_response = planner.run(f"Topic: {query}\nReference Links:\n{links_text}\nWord Limit:{word_limit}", stream=False)
                                
                                writer_response = writer.run(f"Topic: {query}\nWord Limit:{word_limit}\nPlanner outline and text:{planner_response}", stream=False)
                                
                                editor_response = editor.run(f"Topic: {query}\nWord Limit:{word_limit}\nWriter's article draft:{writer_response}", stream=False)

                                # Get the response from the assistant
                                response = publisher.run(editor_response, stream=False)
                                # response = editor_response
                                print("\nPLANNER METRICS: \n", planner.llm.metrics)
                                print("\nWRITER METRICS: \n", writer.llm.metrics)
                                print("\nEDITOR METRICS: \n", editor.llm.metrics)
                                print("\nPUBLISHER METRICS: \n", publisher.llm.metrics,"\n\n")
                                # plan = Planner()
                                # print("PLANNER INSTRUCTIONS: ", plan.instructions)
                                # write= Writer()
                                # print("\nWRITER INSTRUCTIONS: ", write.instructions)
                                # edit = Editor()
                                # print("\nEDITOR INSTRUCTIONS: ", edit.instructions)
                                # pub = Publisher()
                                # print("\nPUBLISHER INSTRUCTIONS: ", pub.instructions)
                        else:
                            st.error("Please provide a topic to write an article on.")
                
        with col2:
            with st.container(border=True):
                if not response == "":
                    st.markdown(response)
                    st_copy_to_clipboard(response)
                else:
                    with st.container():
                        st.image("stock.png", width=350, caption="Your generated article will be displayed here.")
                        
    with tab2:
        get_config()
        
    with tab3:
        getOutput_drafts(planner_response, writer_response, editor_response)
                
spacer_left, form, spacer_right = st.columns([1,1,1], vertical_alignment="bottom")

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_app(FIXED_API_KEY)
    else:
        with form:
            with st.container(border=True):
                st.title("Login")

                # Create login form
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                if st.button("Login", use_container_width=True):
                    if login(username, password):
                        st.session_state.logged_in = True
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
    
if __name__ == "__main__":
    main()

