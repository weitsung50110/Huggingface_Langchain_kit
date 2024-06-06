# Wikipedia # https://python.langchain.com/v0.2/docs/integrations/tools/wikipedia/
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

print(wikipedia.run("Jolin tsai"))
