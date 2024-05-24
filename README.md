## diffuser.py
The model I used is called [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)，<br />
diffuser.py的運行指令如下

    python diffuser.py --output male_teenager2.png --prompt "Taiwanese handsome boy"

* --output代表輸出位置
* --prompt請打你想要生成的圖片形容文字

下面是我使用3種不同prompt產生的結果

    "a Taiwanese handsome boy with blonde hair"
    "a Japanese handsome boy with blonde hair"
    "a Korean handsome boy with blonde hair"

![](https://github.com/weitsung50110/Huggingface_Langchain_kit/blob/master/example_pics/happy_boy0000.png)


    "a cartoon of Taiwanese boy"
    "a cartoon of Japanese boy"
    "a cartoon of Korean boy"

![](https://github.com/weitsung50110/Huggingface_Langchain_kit/blob/master/example_pics/happy_boy00.png)

    "a beautiful japanese girl at the age around 17 in the '80s"
    
![](https://github.com/weitsung50110/Huggingface_Langchain_kit/blob/master/example_pics/happy_boy000.png)

    "a handsome japanese boy at the age around 17 in the '90s"

![](https://github.com/weitsung50110/Huggingface_Langchain_kit/blob/master/example_pics/happy_boy00000.png)

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


## RAG_workflow
![](https://github.com/weitsung50110/Huggingface_Langchain_kit/blob/master/example_pics/RAG_workflow.png)
RAG運作圖參考自 [使用 LangChain 在 HuggingFace 文档上构建高级 RAG](https://huggingface.co/learn/cookbook/zh-CN/advanced_rag)

![](https://github.com/weitsung50110/Huggingface_Langchain_kit/blob/master/example_pics/rag2.png)

## langchain_rag_doc.py
程式碼參考自 [LangChain 怎麼玩？ Retrieval 篇，來做個聊天機器人(ChatBot)吧](https://myapollo.com.tw/blog/langchain-tutorial-retrieval/)

#### chunk_size (塊大小) <br />
定義: 每個分割塊的大小，以字符數量為單位。<br />
作用: 決定每個文本塊包含多少字符。

##### chunk_overlap (塊重疊) <br />
定義: 相鄰文本塊之間重疊的字符數量。<br />
作用: 確保每個分割後的文本塊之間有一些重疊部分，以保證連貫性和上下文不丟失。

**為什麼需要 chunk_overlap?**<br />
在自然語言處理和其他文本分析任務中，連貫性和上下文信息非常重要。<br />
通過設置塊重疊部分，我們可以確保每個分割後的文本塊仍然包含足夠的上下文信息，避免因切割造成的信息丟失或語義斷裂。

    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5) 
    documents = text_splitter.split_documents(docs)

## langchain_rag_pdf.py

#### 'context': context
如果把context註解掉的話，程式也可以RUN，但留著` 'context': context`看起來會比較直觀。

    # context = []
    input_text = input('>>> ')
    while input_text.lower() != 'bye':
        response = retrieval_chain.invoke({
            'input': input_text,
            # 'context': context
        })
        print(response['answer'])
        # context = response['context']
        input_text = input('>>> ')

#### print(response) 
print(response) 也就是把retrieval_chain.invoke給print出來，可以發現長這樣

    {'input': '這篇pdf在說什麼？', 'context': [Document(page_content='全部的PDF文字都會顯示在這裡', 
    metadata={'source': 'weibert.pdf', 'page': 0})], 'answer': 'LLM給的答案會顯示在這裡.'}

由此可以看出response裡面會分別有`response['input']`、`response['context']`、`response['metadata']`、`response['answer']`

metadata={'source': 'weibert.pdf', 'page': 0})] 裡面的page代表LLM在第幾頁PDF找到關聯的資料。

**若我們把程式碼改成如下，把response和response['context']給print出來~**

    while input_text.lower() != 'bye':
        response = retrieval_chain.invoke({
            'input': input_text,
            'context': context
        })
        print(response['answer'])
        context = response['context']
        print("-------------------")
        print(response)
        print("-------------------")
        print(response['context'])
        input_text = input('>>> ')

可以看到結果如下，裡面會分別有`response['input']`、`response['context']`、`response['metadata']`、`response['answer']` <br />
    
    root@4be643ba6a94:/app# python3 langchain_rag_doc.py
    >>> 這篇pdf在說什麼？
    🤔
    
    這篇 pdf 是介紹《哈利波特》世界中的魔法藥物，包括吐真劑（Veritaserum）、變身水（Polyjuice Potion）和福來福喜（Felix Felicix）。這些藥物的特徵和使用方法都有所介紹。
    -------------------
    {'input': '這篇pdf在說什麼？', 'context': [Document(page_content='吐真劑（Veritaserum）出自《火盃的考驗》，特徵為像水一樣清澈無味，
    使用者只要加入三滴，就能強迫飲用者說出真相。它是現存最強大的吐實魔藥，在《哈利波特》的虛構世界觀中受英國魔法部嚴格控管。J·K·羅琳表示，
    吐真劑最適合用在毫無戒心、易受傷害、缺乏自保技能的人身上，有些巫師能使用鎖心術等方式保護自己免受吐真劑影響。'), 
    Document(page_content='變身水（Polyjuice Potion）可變成其他人的樣貌。不可拿來變身成動物，也對動物產生不了效果（包括半人半動物的生物），
    誤用動物毛髮的話，則會變成動物的容貌。'), Document(page_content='福來福喜（Felix Felicix）出自《混血王子》，
    是一種稀有而且難以調製的金色魔藥，能夠給予飲用者好運。魔藥的效果消失之前，飲用者的所有努力都會成功。假如飲用過量，
    會導致頭暈、魯莽和危險的過度自信，甚至成為劇毒。')], 'answer': '🤔\n\n這篇 pdf 是介紹《哈利波特》世界中的魔法藥物，
    包括吐真劑（Veritaserum）、變身水（Polyjuice Potion）和福來福喜（Felix Felicix）。這些藥物的特徵和使用方法都有所介紹。'}
    -------------------
    [Document(page_content='吐真劑（Veritaserum）出自《火盃的考驗》，特徵為像水一樣清澈無味，使用者只要加入三滴，
    就能強迫飲用者說出真相。它是現存最強大的吐實魔藥，在《哈利波特》的虛構世界觀中受英國魔法部嚴格控管。J·K·羅琳表示，
    吐真劑最適合用在毫無戒心、易受傷害、缺乏自保技能的人身上，有些巫師能使用鎖心術等方式保護自己免受吐真劑影響。'), 
    Document(page_content='變身水（Polyjuice Potion）可變成其他人的樣貌。不可拿來變身成動物，也對動物產生不了效果（包括半人半動物的生物），
    誤用動物毛髮的話，則會變成動物的容貌。'), Document(page_content='福來福喜（Felix Felicix）出自《混血王子》，
    是一種稀有而且難以調製的金色魔藥，能夠給予飲用者好運。魔藥的效果消失之前，飲用者的所有努力都會成功。假如飲用過量，會導致頭暈、魯莽和危險的過度自信，甚至成為劇毒。')]

但因為我這個範例是在langchain_rag_doc.py用Document建置的，所以沒有`response['metadata']` <br />
相反的在`response['context']`有三個page_content，是我在langchain_rag_doc.py中一開始就有匯入進去的

    docs = [
        Document(page_content='變身水（Polyjuice Potion）可變成其他人的樣貌。不可拿來變身成動物，也對動物產生不了效果（包括半人半動物的生物），誤用動物毛髮的話，則會變成動物的容貌。'),
        Document(page_content='吐真劑（Veritaserum）出自《火盃的考驗》，特徵為像水一樣清澈無味，使用者只要加入三滴，就能強迫飲用者說出真相。它是現存最強大的吐實魔藥，在《哈利波特》的虛構世界觀中受英國魔法部嚴格控管。J·K·羅琳表示，吐真劑最適合用在毫無戒心、易受傷害、缺乏自保技能的人身上，有些巫師能使用鎖心術等方式保護自己免受吐真劑影響。'),
        Document(page_content='福來福喜（Felix Felicix）出自《混血王子》，是一種稀有而且難以調製的金色魔藥，能夠給予飲用者好運。魔藥的效果消失之前，飲用者的所有努力都會成功。假如飲用過量，會導致頭暈、魯莽和危險的過度自信，甚至成為劇毒。'),
    ]

## langchain_rag_Conversation_Retrieval_Chain.py
我有把chat_history給印出來，可以看到隨著對話越來越長，**_我們的HumanMessage和AIMessage也越來越多_**，<br />
因為記錄下了每一次與LLM的問與答~

    root@4be643ba6a94:/app# python3 langchain_rag_Conversation_Retrieval_Chain.py
    
    >>> i'm 15 years old , study at luna high scholl
    Nice to meet you, Amo! I'd be happy to help answer your questions. You're 15 and studying at Luna High School, right? What's on your mind that you'd like to talk about or ask me? Go ahead!
    --------------------------
    [HumanMessage(content="i'm 15 years old , study at luna high scholl"), AIMessage(content="Nice to meet you, Amo! I'd be happy to help answer your questions. You're 15 and studying at Luna High School, right? What's on your mind that you'd like to talk about or ask me? Go ahead!")]
    >>> i have two friends whoese name are lala and nono
    Nice to meet you too, Amo!
    
    So, you have two friends named Lala and Nono. That's great! Having good friends can make a big difference in your life.
    
    Is there anything specific you'd like to talk about or ask me about your friendships with Lala and Nono? For example, are they part of the same group at school, do you guys have similar interests, or is there something else that makes them special to you?
    --------------------------
    [HumanMessage(content="i'm 15 years old , study at luna high scholl"), AIMessage(content="Nice to meet you, Amo! I'd be happy to help answer your questions. You're 15 and studying at Luna High School, right? What's on your mind that you'd like to talk about or ask me? Go ahead!"), HumanMessage(content='i have two friends whoese name are lala and nono'), AIMessage(content="Nice to meet you too, Amo!\n\nSo, you have two friends named Lala and Nono. That's great! Having good friends can make a big difference in your life.\n\nIs there anything specific you'd like to talk about or ask me about your friendships with Lala and Nono? For example, are they part of the same group at school, do you guys have similar interests, or is there something else that makes them special to you?")]
    >>> do you remember my friends name?
    I remember! You mentioned that your two friends' names are Lala and Nono. I'm glad I got to learn their names from our conversation!
    --------------------------
    [HumanMessage(content="i'm 15 years old , study at luna high scholl"), AIMessage(content="Nice to meet you, Amo! I'd be happy to help answer your questions. You're 15 and studying at Luna High School, right? What's on your mind that you'd like to talk about or ask me? Go ahead!"), HumanMessage(content='i have two friends whoese name are lala and nono'), AIMessage(content="Nice to meet you too, Amo!\n\nSo, you have two friends named Lala and Nono. That's great! Having good friends can make a big difference in your life.\n\nIs there anything specific you'd like to talk about or ask me about your friendships with Lala and Nono? For example, are they part of the same group at school, do you guys have similar interests, or is there something else that makes them special to you?"), HumanMessage(content='do you remember my friends name?'), AIMessage(content="I remember! You mentioned that your two friends' names are Lala and Nono. I'm glad I got to learn their names from our conversation!")]
