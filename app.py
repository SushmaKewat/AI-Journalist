import streamlit as st
import os
from dotenv import load_dotenv
from textwrap import dedent
from phi.assistant import Assistant
from phi.tools.newspaper_toolkit import NewspaperToolkit
from phi.llm.openai import OpenAIChat
from st_copy_to_clipboard import st_copy_to_clipboard
import base64

load_dotenv()

st.set_page_config(page_title="AI Journalist", page_icon="🗞️", layout="wide")

def getlogo():
    with open("logo.png", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return st.markdown(
        f"""
        <div style="position:fixed;
        display:flex;
        align-items:center;
        top:2%;
        z-index:10;
        margin-left:auto;">
            <img src="data:image/webp;base64,{data}" width="150" height="60">
        </div>
        """,
        unsafe_allow_html=True,
    )
getlogo()

# Dummy credentials
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Fixed API key (replace with your actual API key)
FIXED_API_KEY = os.getenv("OPENAI_API_KEY")

st.markdown("""
            <style>
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            
            header[data-testid="stHeader"] {
                background:transparent;
            }
            div[class="block-container st-emotion-cache-1jicfl2 ea3mdgi5"] {
                margin-top:5%;
                padding-top:5%;
            }
            
            h1[id="d88c5d7a"], div[class="st-emotion-cache-fmhvvr e1nzilvr4"] > p {
                text-align:center;
            }
            
            div[class="st-emotion-cache-1v0mbdj e115fcil1"] {
                display: flex;
                flex-direction:column;
                height:auto;
                padding:0.7%;
                margin:auto;
                justify-content: center;
                align-items: center;
            }
            </style>
            """, unsafe_allow_html=True)
# Create a simple login function
def login(username, password):
    return username == USERNAME and password == PASSWORD

article_guidelines = [
    "Inverted Pyramid: This is how you should organize your story. That means the most fundamental, important information (the “base” of the pyramid) goes up at the top, and information that is less crucial goes further down in the story. To figure out what your base is, think about the five Ws: Who, What, When, Where, and Why, as well as the crux of the story.",
    "Lead: The start of a news story should present the most compelling information.",
    "Fact (Not Opinion) and Attribution: Newswriting traditionally doesn’t express opinion unless it’s attributed to a source.",
    "Identification: A person’s full first name or both initials should be used on first reference—not just a single initial. It shouldn’t be assumed that every reader knows who the person is; he or she should be identified in a way that’s relevant to the article.",
    "Short Paragraphs: In newswriting, paragraphs are kept short for punchiness and appearance.",
    "Headlines: Headlines should be short and preferably snappy. They should come out of information in the body of the text and not present new information."
]

def main_app(api_key):
    with st.container():
        st.title("AI Journalist🗞️")
        st.caption("Generate high-quality articles with AI Journalist by researching, writing, and editing articles using GPT-4o.")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    writer = Assistant(
        name="Writer",
        role="Retrieve text from URLs and write high-quality article",
        llm=OpenAIChat(model="gpt-4o", api_key=api_key),
        description=dedent(
            f"""
        You are a senior writer with a 20+ years of experience at the New York Times.
        When writing an article you follow the guidelines mentioned in {article_guidelines}.
        Given a topic and a list of URLs,
        your goal is to write a high-quality NYT-worthy article on the topic using the information from the provided links.
        If no links are provided use your knowledge to curate the article.
        """
        ),
        instructions=[
            "Given a topic and a list of URLs(may or may not be given), read each article in depth.",
            "Collect all the major points relevant to the context."
            "Write a high-quality NYT-worthy article on the topic within the word limit. Do not exceed the given word limit.",
            f"Curate the article based on the guidelines in the {article_guidelines}."
            "Retain the original entities and mention them in the articles such as people names, places, dates, amounts, quotes and other significant attributes.",
            "Write in proper headings/sections and subheadings/subsections.",
            "Ensure you provide a nuanced and balanced opinion, quoting facts where possible.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
            "At the end of the article, Create a sources list of each result you cited, with the article name, author, and link."
        ],
        tools=[NewspaperToolkit()],
        add_datetime_to_instructions=True,
        add_chat_history_to_prompt=True,
        num_history_messages=3,
    )

    editor = Assistant(
        name="Editor",
        llm=OpenAIChat(model="gpt-4o", api_key=api_key),
        team=[writer],
        description="You are a senior NYT editor. Given a topic, your goal is to write a NYT-worthy article.",
        instructions=[
            "Given a topic, URLs and word limit, pass the description of the topic and URLs to the writer to get a draft of the article.",
            f"Format the article based on the guidelines in the {article_guidelines}."
            "Edit, proofread, and refine the article to ensure it meets the high standards of the New York Times.",
            "The article should be extremely articulate and well-written.",
            "Focus on clarity, coherence, and overall quality.",
            "Ensure the article is engaging and informative.",
            "Remember: you are the final gatekeeper before the article is published. Do not add extra comments from your side at the end of the article.",
        ],
        add_datetime_to_instructions=True,
        markdown=True,
    )

    response = ""
    # Input field for the report query
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.header("Input & Configuration")
            query = st.text_input("What do you want the AI journalist to write an article on?", placeholder="E.g: Emergence of AI and LLMs.")
            
            word_limit = st.slider("How long should be your article?", min_value=250, max_value=1500, step=50, key="word_limit")

            use_links = st.radio("Do you want to provide reference links?", ("No", "Yes"))

            links = []
            if use_links == "Yes":
                    num_links = st.number_input("How many links do you want to provide?",placeholder="Enter the number of links that you want to use", min_value=1, max_value=5, step=1, key="number_of_links", help="These links will be used to curate your news article.")
                    for i in range(num_links):
                        link = st.text_input(f"Enter reference link {i+1}", key=f"link_{i+1}")
                        links.append(link)
           
            if use_links == "No" or (use_links == "Yes" and all(links)):
                    if st.button("Generate Article"):
                        if query:
                            with st.spinner("Good things take time, and we're making sure it's perfect for you!"):
                        # Prepare the content for the writer
                                links_text = "\n".join(links) if links else "No reference links provided."
                                writer_instructions = f"Topic: {query}\nReference Links:\n{links_text}\nWord Limit:{word_limit}"

                        # Get the response from the assistant
                                response = editor.run(writer_instructions, stream=False)
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

