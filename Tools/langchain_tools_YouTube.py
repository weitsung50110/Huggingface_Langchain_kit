# YouTubeSearchTool Search # https://python.langchain.com/v0.2/docs/integrations/tools/youtube/
from langchain_community.tools import YouTubeSearchTool

tool = YouTubeSearchTool()

print(tool.run("五月天")) #預設只會返回2筆

# 可以指定你要返回幾筆搜尋結果
print(tool.run("五月天, 5"))
