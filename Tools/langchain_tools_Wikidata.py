# Wikidata # https://python.langchain.com/v0.2/docs/integrations/tools/wikidata/
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

print(wikidata.run("林依晨"))

