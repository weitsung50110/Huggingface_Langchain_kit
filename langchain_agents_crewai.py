from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# 初始化語言模型和嵌入模型
ollama_llm = Ollama(model='llama3')
embeddings = OllamaEmbeddings()

from langchain.tools import DuckDuckGoSearchRun
from crewai_tools import tool

search_tool = DuckDuckGoSearchRun()

@tool("DuckDuckGoSearch")
def search(search_query:str):
    """search the web for information for the given topic"""
    return DuckDuckGoSearchRun().run(search_query)

researcher = Agent(
  role='Researcher',
  goal='Search the internet for the information requested',
  backstory="""
  You are a researcher. Using the information in the task, you find out some of the most popular facts about the topic along with some of the trending aspects.
  You provide a lot of information thereby allowing a choice in the content selected for the final blog.
  """,
  verbose=True,
  allow_delegation=False,
  tools=[search],
  llm=ollama_llm
)

writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on a set of information provided by the researcher.',
  backstory="""You are a writer known for your humorous but informative way of explaining.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  llm=ollama_llm
)

task1 = Task(
  description="""Research about open source LLMs vs closed source LLMs.
    Your final answer MUST be a full analysis report.""",
    agent=researcher,
    expected_output="A comprehensive analysis report detailing the differences, advantages, and disadvantages of open-source and closed-source LLMs."
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant facts and differences between open-source LLMs and closed-source LLMs.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, and avoid complex words so it doesn't sound like AI.
  Your final answer for the blog post should be between 100 words.""",
  agent=writer,
  expected_output="A blog post of approximately 100 words that highlights the key differences between open-source and closed-source LLMs, written in an engaging and accessible style."
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 for different logging levels
)

# Get your crew to work!
result = crew.kickoff()

# https://www.linkedin.com/pulse/beginners-guide-creating-ai-agents-langchain-vijaykumar-kartha-gwrrc?trk=portfolio_article-card_title
# https://github.com/premthomas/Ollama-and-Agents
# https://github.com/joaomdmoura/crewAI/issues/316