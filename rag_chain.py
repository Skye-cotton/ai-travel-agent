import os
from langchain_chroma import Chroma
from langchain_zhipu import ChatZhipuAI,ZhipuAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

def format_docs(docs):
    """将捞出来的多个 Document 块用双换行拼接成一段纯文本背景"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # 初始化数据库，连接已有数据库
    embeddings = ZhipuAIEmbeddings(
        model="embedding-3",
        zhipuai_api_key = os.getenv('ZHIPUAI_API_KEY')
    )

    # 加载本地持久化数据库
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    # 转换为 LangChain 标准检索器（Retriever），并指定捞出最相关的 2 块数据 (k=2)
    retriever = vector_store.as_retriever(search_kwargs={"k":2})
    # 打造防御性超强的 RAG Prompt 模板（防止大模型瞎编）
    template = """你是一位严格、诚实的专业旅游数据核验员。请严格根据下面提供的【真实小红书避坑线索】来回答用户的提问。

        【终极硬性防御规则】
        1. 你【只能】使用【真实小红书避坑线索】中明确提及的事实。
        2. 如果线索中没有包含能直接回答用户提问的完整答案，你【必须且只能】逐字回复：“非常抱歉，根据目前的私有攻略库未能找到相关避坑提示。”
        3. 严禁使用你自身的预训练知识进行任何补充、推理、建议或推荐！哪怕是一句话、一个酒店名字也绝对不允许输出！
        4. 违反上述规则将导致系统安全崩溃。

        【真实小红书避坑线索】：
        {context}

        用户的问题：{question}
        请给出符合规则的回答："""

    prompt = ChatPromptTemplate.from_template(template)

    # 初始化大模型底座
    llm = ChatZhipuAI(
        model='glm-4-plus',
        temperature=0.1,
        zhipuai_api_key=os.getenv('ZHIPUAI_API_KEY')
    )

    # 组装工业级LCEL经典RAG链条
    rag_chain = (
        {
            "context": retriever | format_docs, # 提问先扔给检索器，捞出文档并格式化
            "question": RunnablePassthrough() # 提问原封不动传给下一个环节
        }
        | prompt 
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    print("====正在初始化智能RAG问答链=====")
    chain = get_rag_chain()

    print("\n 问答系统已就绪！请输入您想咨询的杭州旅游问题（输入 q 退出）：")
    while True:
        user_query = input("\n 用户提问").strip()
        if user_query.lower() in ['q','exit']:
            print("再见")
            break
        if not user_query:
            continue

        print("正在检索本地知识库并进行深度思考...")
        # 驱动RAG链条
        response = chain.invoke(user_query)
        print(f"AI 回答：\n{response}")