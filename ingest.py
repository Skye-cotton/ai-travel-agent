import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 新增导入：embedding 模型和 chroma 向量数据库
from langchain_zhipu import ZhipuAIEmbeddings
from langchain_chroma import Chroma

# 本地向量数据库存储路径
PERSIST_DIRECTORY= "chroma_db"


def clean_xhs_text(raw_text: str) -> str:
    """
    针对小红书文本的高级预处理函数
    清洗无用噪声，但保留核心地标、价格和逻辑词
    """
    # 1. 过滤掉无用文字
    text = re.sub(r"家人们谁懂啊！？", "",raw_text)
    text = re.sub(r"大数据请把这条推送给.*！", "",text)
    text = re.sub(r"家人们谁懂啊宝子们，?", "",text)
    # 2. 将连续的多个换行符压缩为双换行，保证切片器能识别出明显的段落
    text = re.sub(r"\n\s*\n","\n\n", text)

    return text.strip()

def run_ingestion_pipeline():
    print("========开始执行小红书文本清洗与切片流水线======")

    file_path = "data/hangzhou_xiaohongshu.txt"
    if not os.path.exists(file_path):
        print(f"未找到源文件: {file_path}，请先创建数据。")
        return
    # 1. 读取原始数据
    with open(file_path,"r", encoding="utf-8") as f:
        raw_content = f.read()

    # 2. 执行文本清洗
    cleaned_content = clean_xhs_text(raw_content)
    print("文本清洗完成，无用噪声已过滤。")

    # 3. 初始化智能切片器
    # Chunk_size 设为 300 字左右，因为小红书一个景点的核心干货差不多这个长度
    # Overlap 设为 50 字，确保跨段落的上下文不会丢失
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 300,
        chunk_overlap=50,
        length_function= len,
        separators=["\n\n","\n","。", "！", " "]
    )
    # 执行切片
    chunks = text_splitter.create_documents([cleaned_content])
    print(f"🧩 成功将非结构化攻略切分为 [ {len(chunks)} ] 个高质量核心文本块！\n")

    # 4. 初始化智谱 Embedding 模型
    # 工业界标准：使用统一的 API Key 来实例化向量模型
    embeddings = ZhipuAIEmbeddings(
        model='embedding-3',
        zhipuai_api_key ='ZHIPUAI_API_KEY'
    )
    print(" 智谱 Embedding-3 向量引擎初始化成功...")

    # 向量化并持久化写入chromaDB
    print("💾 正在调用接口进行向量转换并写入本地知识库...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"小红书深度知识库已成功持久化构建！")
    print(f"知识库已安全保存在本地目录: ./{PERSIST_DIRECTORY}/\n")

    # 搜索测试
    print("[现场验证] 模拟向量相似度检索测试中...")
    test_query = "灵隐寺防骗"
    # 捞出最相似的Top1切片
    docs = vector_store.similarity_search(test_query, k=1)

    if docs:
        print("匹配成功，最相关的攻略如下：")
        print(docs[0].page_content)
    else:
        print("未检索到相关内容")

    # # 5. 打印切片结果，观察边界是否完美;enumerate可以拿到索引和值 for in只能拿到值
    # for idx,chunk in enumerate(chunks):
    #     print(f"---- Chunk {idx + 1} ----")
    #     print(chunk.page_content)
    #     print("-" * 30)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    run_ingestion_pipeline()