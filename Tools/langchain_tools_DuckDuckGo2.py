# 會產生好幾個snippet和title片段
# DuckDuckGo Search # https://python.langchain.com/v0.2/docs/integrations/tools/ddg/
from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()

print(search.run("黃仁勳"))


#https://www.toolify.ai/tw/ai-news-tw/%E4%BD%BF%E7%94%A8langchain%E8%88%87duckduckgowikipedia%E5%92%8Cpython-repl%E5%B7%A5%E5%85%B7-1096402