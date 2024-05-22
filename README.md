## diffuser.py
The model I used is called [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)，<br />
diffuser.py的運行指令如下

    python diffuser.py --output male_teenager2.png --prompt "chinese handsome boy"

* --output代表輸出位置
* --prompt請打你想要生成的圖片形容文字

![](https://github.com/weitsung50110/Huggingface_Langchain_kit/blob/master/example_pics/road.png)

![](https://github.com/weitsung50110/Huggingface_Langchain_kit/blob/master/example_pics/male.png)


## langchain_sys_SEOtitle_article_generate.py

本檔案是使用llama2和llama3來執行，切換的程式碼如下

    llm = Ollama(model='llama3')
    
* 如果加上了CallbackManager就可以即時看到llm生成的文字，
* 若沒加CallbackManager則是要等到llm把文字全部生成完成後 才會顯示出來

        model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])

執行程式語法如下，當看到>>>時，請輸入想要生成的文章概念

    root@4be643ba6a94:/app# python3 langchain_sys_SEOtitle_article_generate.py
    >>> love

輸入love以後，LLM會幫你生成文章，並且給你SEO標題
    **Title:** The Power of Unconditional Love: How It Can Transform Your Life
    
    **Article:**
    
    Unconditional love has the power to transform our lives in profound ways. When we love someone without condition, we open ourselves up to a deeper connection and  
    understanding. This type of love is not based on what someone does for us, but rather who they are as a person. 
    It's a choice to accept and cherish them just as they are. By practicing unconditional love, we can build stronger relationships, 
    cultivate empathy and compassion, and even improve our mental health. In a world that often values conditionality, 
    it's essential to remember the transformative power of loving someone without expectation or attachment.
    
    **SEO Title:** "The Transformative Power of Unconditional Love: Boosting Mental Health and Building Stronger Relationships"

    Keywords: unconditional love, transformative power, mental health, relationships, self-acceptance.
