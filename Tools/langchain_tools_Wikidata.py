# Wikidata # https://python.langchain.com/v0.2/docs/integrations/tools/wikidata/
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

print(wikidata.run("林依晨"))



#https://www.toolify.ai/tw/ai-news-tw/%E4%BD%BF%E7%94%A8langchain%E8%88%87duckduckgowikipedia%E5%92%8Cpython-repl%E5%B7%A5%E5%85%B7-1096402