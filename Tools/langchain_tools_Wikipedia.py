# Wikipedia # https://python.langchain.com/v0.2/docs/integrations/tools/wikipedia/
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

print(wikipedia.run("Jolin tsai"))




#https://www.toolify.ai/tw/ai-news-tw/%E4%BD%BF%E7%94%A8langchain%E8%88%87duckduckgowikipedia%E5%92%8Cpython-repl%E5%B7%A5%E5%85%B7-1096402