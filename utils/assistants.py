import os
from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.tools.newspaper4k import Newspaper4k

from utils.guidelines import article_guidelines

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

class JournalistTeam:
    assistant = None
    def __init__(self, name, role, description, instructions, team=None, llm=None, **kwargs):
        self.name = name
        self.role = role
        self.team = team
        self.llm = llm
        self.__description = description
        self.__instructions = instructions
        self.tools = kwargs.get('tools', [])
        
    def create_assistant(self) -> Assistant:
        assistant = Assistant(
            name=self.name,
            role=self.role,
            llm=self.llm,
            description=self.__description,
            instructions=self.__instructions,
            tools=self.tools,
            show_tool_calls=True,
            debug_mode=True,
            prevent_hallucinations=True
        )
        return assistant

    # def run_assistant(self, instructions: str) -> str:
    #     return self.assistant.run(instructions)
        
    def __repr__(self):
        return f"Assistant name:{self.name}\n\nAssistant role:{self.role}\n\nLLM model:{self.llm}"


class Publisher(JournalistTeam):
    name = "Publisher"
    
    role = "Publishing the articles"
    
    __description = [
        "You are a superviser/advisor of writer and editor.",
        "You are responsible for publishing the article to the user.",
        "You play an important role in the development and dissemination of content.",
        "You are the key decision-maker in the production process, overseeing all stages from concept selection to final publication."
    ]
    
    __instructions = [
        "Given a topic and a list of URLs, your goal is to publish a high-quality NYT-worthy article on the topic using the information from the provided links.",
        # "You have to delegate the planning, writing and editing of the article to the planner, writer and editor respectively.",
        # "Ensure that every assisstant completes their assigned tasks.",
        "Get the final article from the editor.",
        # "DO NOT POST THE WORD COUNT OR TASK DELEGATION.",
        "The final output must be only the article."
    ]
    
    def __init__(self):
        super().__init__(
            name=self.name,
            role=self.role,
            description=" ".join(self.__description),
            instructions=self.__instructions,
            #llm=OpenAIChat(model="gpt-4o", api_key=API_KEY, temperature=0),
        )
    
    @property
    def instructions(self):
        """Get the instructions for the Planner."""
        return self.__instructions

    
    @instructions.setter
    def instructions(self, instructions):
        self.__instructions = instructions
    
    def reset_instructions(self):
        self.__instructions = self.__instructions
          
    def __repr__(self):
        return super().__repr__()

class Planner(JournalistTeam):
    name = "Planner"
    role = "Plan the article outline and get the content from the given links."
    
    __description = [
        "You are a professional news article planner.",
        "Given a topic you need to formulate an article outline.",
        "Based on the formulated outline, you need to get the content from the given links.",
    ]
    
    __instructions = [
        "Get the topic and the list of URLs from the user.",
        "Extract text from the given URLs.",
        "Based on the content of the extracted text, create a highly comprehensive and well formulated outline for the article.",
        "Pass all this information to the writer along with the links given by user."
    ]
    
    _tools = [Newspaper4k(include_summary=True)]
    
    def __init__(self):
        if self.__description is None:
            raise ValueError("_description cannot be None")
        if self.__instructions is None:
            raise ValueError("_instructions cannot be None")

        self.__description = " ".join(self.__description)
        self.description = self.__description
        self.instructions = self.__instructions
        super().__init__(
            name=self.name,
            role=self.role,
            description=self.description,
            instructions=self.instructions,
            llm=OpenAIChat(model="gpt-4o", api_key=API_KEY, temperature=0),
            tools=self._tools,
        )
        
    @property
    def instructions(self):
        """Get the instructions for the Planner."""
        return self.__instructions

    
    @instructions.setter
    def instructions(self, instructions):
        self.__instructions = instructions
    
    def reset_instructions(self):
        self.__instructions = self.__instructions
        
    def __repr____(self):
        return super().__repr__()
  
class Writer(JournalistTeam):
    name = "Writer"
    
    role = "Write high quality New York Times-worthy news articles."
    
    __description = [
        "You are a senior writer at New York Times.",
        "Given a topic and text content from Planner, your goal is to write a high-quaity NYT-worthy article on the topic.",
        "Finally send the article draft to the Editor."
    ]
    
    __instructions = [
        "Get the topic, text and word limit from the Planner.",
        "Write an article using the content and follow the outline for the article curated by the Planner.",
        f"The article should be written based on the guidelines as follows: {article_guidelines}",
        "Always retain the entities involved in the news, such as, people names, places, dates, numbers, amounts, quotes, etc.",
        "Ensure that the article has headings & sub-headings and is well-written and well-structured.",
        "Always provide a nuanced and balanced opinion, quoting facts where possible.",
        "Focus on clarity, coherence and overall quality.",
        "Never make up facts or plagiarize. Always provide proper attribution.",
        "At the end of each article, Create a sources list of each result you cited, with the article name, author, and link."
    ]
    def __init__(self):
        self.name = "Writer"
        self.role = "Write high quality New York Times-worthy news articles."
        self.description = " ".join(self.__description)
        self.instructions = self.__instructions
        self.llm = OpenAIChat(model="gpt-4o", api_key=API_KEY, temperature=0)
        super().__init__(self.name, self.role, self.description, self.instructions, self.llm)
    
    @property
    def instructions(self):
        """Get the instructions for the Planner."""
        return self.__instructions

    
    @instructions.setter
    def instructions(self, instructions):
        self.__instructions = instructions
    
    def reset_instructions(self):
        self.__instructions = self.__instructions
          
    def __repr__(self):
        return super().__repr__()

class Editor(JournalistTeam): 
    name = "Editor"
    
    role = "Get article draft from writer, proofread and edit it as per NYT standards."
    
    __description = [
        "You are a senior editor at New York Times.",
        "Your goal is to edit the article draft from the Writer.",
    ]  
    
    __instructions = [
        "Get the article draft from the Writer.",
        "Proofread the article using the tools to ensure it meets the high standards of the New York Times.",
        "Ensure the article has proper headings and subheadings.",
        "Check for facts and citations in the article.",
        "Ensure the original entites are retained in the article and the overall essence of the article is well curated.",
        "The article should be extremely articulate and well-written.",
        "Ensure the article is engaing and informative.",
        "Make sure the article is within the given word limit.",
        "Ensure the links are cited."
    ]
    
    __tools = [Newspaper4k(include_summary=True)]
    
    def __init__(self):
        if self.__description is None:
            raise ValueError("_description cannot be None")
        if self.__instructions is None:
            raise ValueError("_instructions cannot be None")

        self.description = " ".join(self.__description)
        self.instructions = self.__instructions
        super().__init__(self.name, self.role, self.description, self.instructions, OpenAIChat(model="gpt-4o", api_key=API_KEY, temperature=0), tools=self.__tools)
    
    @property
    def instructions(self):
        """Get the instructions for the Planner."""
        return self.__instructions

    
    @instructions.setter
    def instructions(self, instructions):
        self.__instructions = instructions
    
    def reset_instructions(self):
        self.__instructions = self.__instructions
            
    def __repr__(self):
        return super().__repr__()
