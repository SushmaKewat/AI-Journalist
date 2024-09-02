import os
from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.tools.newspaper4k import Newspaper4k

from utils.guidelines import article_guidelines

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

class JournalistTeam:
    def __init__(self, name, role, description, instructions, team=None, llm=None, **kwargs):
        self.name = name
        self.role = role
        self.team = team
        self.llm = llm
        self.description = description
        self.instructions = instructions
        self.tools = kwargs.get('tools', [])
        
    def create_assistant(self) -> Assistant:
        assistant = Assistant(
            name=self.name,
            role=self.role,
            llm=self.llm,
            description=self.description,
            instructions=self.instructions,
            tools=self.tools,
            show_tool_calls=True,
            debug_mode=True,
            prevent_hallucinations=True
        )
        return assistant

    def run_assistant(self, assistant: Assistant) -> str:
        return assistant.run()
        
    def __repr__(self):
        return f"Assistant name:{self.name}\n\nAssistant role:{self.role}\n\nLLM model:{self.llm}"


class Publisher(JournalistTeam):
    name = "Publisher"
    
    role = "Publishing the articles"
    
    _description = [
        "You are a superviser/advisor of writer and editor.",
        "You are responsible for publishing the article to the user.",
        "You play an important role in the development and dissemination of content.",
        "You are the key decision-maker in the production process, overseeing all stages from concept selection to final publication."
    ]
    
    _instructions = [
        "Given a topic and a list of URLs,your goal is to publish a high-quality NYT-worthy article on the topic using the information from the provided links.",
        "You have to delegate the planning, writing and editing of the article to the planner, writer and editor respectively.",
        "Ensure that every assisstant completes their assigned tasks.",
        "Get the final article from the editor and publish it to the user."
    ]
    
    def __init__(self):
        super().__init__(
            name=self.name,
            role=self.role,
            description=" ".join(self._description),
            instructions=self._instructions,
            llm=OpenAIChat(model="gpt-4o", api_key=API_KEY, temperature=0),
        )
        
    def __repr__(self):
        return super().__repr__()

class Planner(JournalistTeam):
    name = "Planner"
    role = "Plan the article outline and get the content from the given links."
    
    _description = [
        "You are a professional news article planner.",
        "Given a topic you need to formulate an article outline.",
        "Based on the formulated outline, you need to get the content from the given links.",
    ]
    
    _instructions = [
        "Get the topic and the list of URLs from the user.",
        "Extract text from the given URLs.",
        "Based on the content of the extracted text, create a highly comprehensive and well formulated outline for the article.",
        "Pass all this information to the writer."
    ]
    
    _tools = [Newspaper4k(include_summary=True)]
    
    def __init__(self):
        if self._description is None:
            raise ValueError("_description cannot be None")
        if self._instructions is None:
            raise ValueError("_instructions cannot be None")

        self._description = " ".join(self._description)
        self.description = self._description
        self.instructions = self._instructions
        super().__init__(
            name=self.name,
            role=self.role,
            description=self.description,
            instructions=self.instructions,
            llm=OpenAIChat(model="gpt-4o", api_key=API_KEY, temperature=0),
            tools=self._tools,
        )
        
        
    def __repr____(self):
        return super().__repr__()
  
class Writer(JournalistTeam):
    name = "Writer"
    
    role = "Write high quality New York Times-worthy news articles."
    
    _description = [
        "You are a senior writer at New York Times.",
        "Given a topic and text content from Planner, your goal is to write a high-quaity NYT-worthy article on the topic.",
        "Finally send the article draft to the Editor."
    ]
    
    _instructions = [
        "Get the topic, text and word limit from the Planner.",
        "Write an article using the content and follow the outline for the article curated by the Planner.",
        f"The article should be written based on the guidelines as follows: {article_guidelines}",
        "Always retain the entities involved in the news, such as, people names, places, dates, numbers, amounts, quotes, etc.",
        "Ensure that the article is well-written and well-structured.",
        "Always provide a nuanced and balanced opinion, quoting facts where possible.",
        "Focus on clarity, coherence and overall quality.",
        "Never make up facts or plagiarize. Always provide proper attribution.",
        "At the end of each article, Create a sources list of each result you cited, with the article name, author, and link."
    ]
    def __init__(self):
        self.name = "Writer"
        self.role = "Write high quality New York Times-worthy news articles."
        self.description = " ".join(self._description)
        self.instructions = self._instructions
        self.llm = OpenAIChat(model="gpt-4o", api_key=API_KEY, temperature=0)
        super().__init__(self.name, self.role, self.description, self.instructions, self.llm)
        
    def __repr__(self):
        return super().__repr__()

class Editor(JournalistTeam): 
    name = "Editor"
    
    role = "Get article draft from writer, proofread and edit it as per NYT standards."
    
    _description = [
        "You are a senior editor at New York Times.",
        "Your goal is to edit the article draft from the Writer.",
    ]  
    
    _instructions = [
        "Get the article draft from the Writer.",
        "Proofread the article to ensure it meets the high standards of the New York Times.",
        "Check for facts and citations in the article.",
        "Ensure the original entites are retained in the article and the overall essence of the article is well curated.",
        "The article should be extremely articulate and well-written.",
        "Ensure the article is engaing and informative.",
        "Make sure the article is within the given word limit."
    ]
    def __init__(self):
        if self._description is None:
            raise ValueError("_description cannot be None")
        if self._instructions is None:
            raise ValueError("_instructions cannot be None")

        self.description = " ".join(self._description)
        self.instructions = self._instructions
        super().__init__(self.name, self.role, self.description, self.instructions, OpenAIChat(model="gpt-4o", api_key=API_KEY, temperature=0))
