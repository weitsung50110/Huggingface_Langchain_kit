# 會產生好幾個snippet和title片段
# DuckDuckGo Search # https://python.langchain.com/v0.2/docs/integrations/tools/ddg/
from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()

print(search.run("黃仁勳"))

