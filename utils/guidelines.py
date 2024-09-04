article_guidelines = [
    "Inverted Pyramid: This is how you should organize your story. That means the most fundamental, important information (the “base” of the pyramid) goes up at the top, and information that is less crucial goes further down in the story.",
    "Lead: The start of a news story should present the most compelling information.",
    "Fact (Not Opinion) and Attribution: State the actual facts, figures, dates and numbers and always provide proper attribution.",
    "Identification: A person’s full first name or both initials should be used on first reference—not just a single initial. It shouldn’t be assumed that every reader knows who the person is; he or she should be identified in a way that’s relevant to the article.",
    "Short Paragraphs: In newswriting, paragraphs are kept short for punchiness and appearance.",
    "Headlines: Headlines should be short and preferably snappy. They should come out of information in the body of the text and not present new information."
    "Conclusion: Always end the article with a proper conclusion heading restating the premise."
]

# article_guidelines = [
#     "Inverted Pyramid: This is how you should organize your story. That means the most fundamental, important information (the “base” of the pyramid) goes up at the top, and information that is less crucial goes further down in the story.",
#     "Lead: The start of a news story should present the most compelling information.",
#     "Fact (Not Opinion) and Attribution: State the actual facts, figures, dates and numbers and always provide proper attribution.",
#     "Identification: A person’s full first name or both initials should be used on first reference—not just a single initial. It shouldn’t be assumed that every reader knows who the person is; he or she should be identified in a way that’s relevant to the article.",
#     "Short Paragraphs: In newswriting, paragraphs are kept short for punchiness and appearance.",
#     "Headlines: Headlines should be short and preferably snappy. They should come out of information in the body of the text and not present new information."
#     "Conclusion: Always end the article with a proper conclusion heading restating the premise."
# ]

# editing_principles = [
#     "Accuracy: Checking and crosschecking names, figures and verifying facts are of utmost importance.",
#     "Attribution: Always attribute the news to the source so that readers can judge its credibility.",
#     "Brevity:  It is telling a story, as it should be, without beating around the bush.",
#     "Readability: The average length of a sentence should not exceed 18 words, which is standard. The best way is to write news stories using simple words, short and simple sentences.",
#     "Explanatory or Analysis—still opinion, but mostly casts new light on ongoing issue.",
#     "The editorial opens with power and closes with purpose. Begin with a premise or strongly worded opinion then wrap up with a conclusion that restates the premise. ",
#     "In the body, provide facts, information and statistics to support your premise. ",
#     " The facts (evidence) should be as complete as possible in the space allowed. Avoid repeating arguments in the body, even if using different language.",
#     "Finish with a conclusion that restates the premise."
# ]

 # writer = Assistant(
    #     name="Writer",
    #     role="Retrieve text from URLs and write high-quality article",
    #     llm=OpenAIChat(model="gpt-4o", api_key=api_key, temperature=0),
    #     description=dedent(
    #         f"""
    #     You are a senior writer with a 20+ years of experience at the New York Times.
    #     Given a topic and a list of URLs,your goal is to write a high-quality NYT-worthy article on the topic using the information from the provided links.
    #     If no links are provided use your knowledge to curate the article.
    #     """
    #     ),
    #     instructions=[
    #         "Write a high-quality NYT-worthy article on the given topic within the word limit. Do not exceed the given word limit.",
    #         "People involved and mentioned in the text, places, dates, numbers, amounts, quotes, etc, all these things should be retained and must be mentioned in the final article."
    #         f"Curate the article based on the guidelines in the {article_guidelines}."
    #         "Write in proper headings/sections and subheadings/subsections.",
    #         "Ensure you provide a nuanced and balanced opinion, quoting facts where possible.",
    #         "Focus on clarity, coherence, and overall quality.",
    #         "Never make up facts or plagiarize. Always provide proper attribution.",
    #         "At the end of the article, Create a sources list of each result you cited, with the article name, author, and link."
    #     ],
    #     tools=[Newspaper4k()],
    #     show_tool_calls=True,
    #     debug_mode=True,
    #     prevent_hallucinations=True,
    # )

    # editor = Assistant(
    #     name="Editor",
    #     team=[writer],
    #     llm=OpenAIChat(model="gpt-4o", api_key=api_key, temperature=0),
    #     role="Get the draft of the article from writer and edit it as per the given instructions.",
    #     description="You are a senior NYT editor. Given a topic, your goal is to write a NYT-worthy article.",
    #     instructions=[
    #         "Given a topic, URLs and word limit, pass the description of the topic and URLs to the writer to get a draft of the article.",
    #         #f"Format the article based on the guidelines in the {editing_principles}."
    #         "Edit, proofread, and refine the article to ensure it meets the high standards of the New York Times.",
    #         "The article should be extremely articulate and well-written.",
    #         "Focus on clarity, coherence, and overall quality.",
    #         "Ensure the article is engaging and informative."
    #     ],
    #     add_datetime_to_instructions=True,
    #     markdown=True,
    #     debug_mode=True
    # )
    