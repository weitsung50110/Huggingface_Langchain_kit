# 只會產生一筆result
# DuckDuckGo Search # https://python.langchain.com/v0.2/docs/integrations/tools/ddg/
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

print(search.run("黃仁勳"))

