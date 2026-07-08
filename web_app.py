import streamlit as st
from agents import (
    extractor_chain, 
    planner_chain, 
    mock_search_places, 
    mock_get_weather, 
    mock_estimate_budget_rules, 
    extractor_parser, 
    planner_parser
)
from models import TravelConstraints, FullTravelPlan
import os
# 引入 RAG 知识库组件
from langchain_chroma import Chroma
from langchain_zhipu import ZhipuAIEmbeddings
from rag_chain import format_docs


# 初始化本地RAG检索器
# @st.cache_resource # 使用 Streamlit 缓存，防止每次刷新页面都重复加载数据库
def init_local_retriever():
    # 检查 session_state 里是否已经初始化过了，如果没有，只初始化一次
    if "global_retriever" not in st.session_state:
        embeddings = ZhipuAIEmbeddings(
            model='embedding-3',
            zhipuai_api_key=os.getenv('ZHIPUAI_API_KEY')
        )
        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        st.session_state.global_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return st.session_state.global_retriever

retriever = init_local_retriever()

# 1. 页面基本配置
st.set_page_config(
    page_title="智能AI旅行管家",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 初始化 Session 状态（长对话记忆核心）
if "messages" not in st.session_state:
    st.session_state.messages = []  # 存储网页聊天气泡历史

if "current_constraints" not in st.session_state:
    st.session_state.current_constraints = None  # 存储结构化约束信息

if "current_plan" not in st.session_state:
    st.session_state.current_plan = None  # 存储最新的完整行程方案

# =================页面左侧栏：状态监视器 =================
with st.sidebar:
    st.title("🗺️ 记忆状态监视器")
    st.write("大模型当前捕获的结构化背景：")
    
    if st.session_state.current_constraints:
        c = st.session_state.current_constraints
        st.success(f"📍 **目的地**: {c.destination or '未提取'}")
        st.info(f"📅 **天数**: {c.duration_days or '未提取'} 天")
        st.warning(f"💰 **预算上限**: {c.budget or '未指定'} 元")
        st.write(f"🏷️ **偏好标签**: {', '.join(c.tags) if c.tags else '无'}")
        st.write(f"🛑 **出行限制**: {', '.join(c.restrictions) if c.restrictions else '无'}")
    else:
        st.caption("⏳ 暂无结构化记忆，请在右侧输入您的旅行想法。")
        
    st.markdown("---")
    if st.button("🗑️ 清空对话记忆", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_constraints = None
        st.session_state.current_plan = None
        st.rerun()

# =================  页面右侧主区：ChatGPT 交互流 =================
st.title("🤖 您的全能智能旅行管家")
st.caption("基于 LangChain + Pydantic 强约束架构 | 实时注入天气与价格风控网关")

# 渲染历史聊天气泡
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 接收用户新输入
if user_input := st.chat_input("您可以说：我想带父母去杭州玩2天，别太累..."):
    # 1. 立即将用户话语上屏
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2. 状态驱动核心逻辑
    with st.chat_message("assistant"):
        with st.spinner("🤖 思考中，请稍候..."):
            
            # 场景 A：新开局，提取约束
            if st.session_state.current_constraints is None:
                try:
                    extracted: TravelConstraints = extractor_chain.invoke({
                        "user_input": user_input,
                        "format_instructions": extractor_parser.get_format_instructions()
                    })
                    st.session_state.current_constraints = extracted
                except Exception as e:
                    st.error(f"约束解析失败，请换个表述试试。错误: {e}")
                    st.stop()

                # 拦截关键信息缺失
                c = st.session_state.current_constraints
                if not c.destination or not c.duration_days:
                    missing = []
                    if not c.destination: missing.append("想去哪个城市")
                    if not c.duration_days: missing.append("计划玩几天")
                    response = f"您的想法很棒！不过能顺便告诉我您{' 和 '.join(missing)}吗？"
                    
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.current_constraints = None  # 重置以便下次继续提取
                    st.stop()
            
            # 场景 B：已有行程，用户提修改意见
            else:
                st.session_state.current_constraints.restrictions.append(f"用户最新修改要求：{user_input}")
                # 【新增修复】尝试用 extractor_chain 提取新输入中包含的结构化字段
                try:
                    new_extracted: TravelConstraints = extractor_chain.invoke({
                        "user_input": user_input,
                        "format_instructions": extractor_parser.get_format_instructions()
                    })
                    # 如果是新的目的地覆盖掉
                    if new_extracted.destination:
                        st.session_state.current_constraints.destination = new_extracted.destination
                    # 如果用户提到了新的天数，进行覆盖
                    if new_extracted.duration_days:
                        st.session_state.current_constraints.duration_days = new_extracted.duration_days
                    # 如果用户提到了新的预算，进行覆盖
                    if new_extracted.budget:
                        st.session_state.current_constraints.budget = new_extracted.budget
                        
                    # 标签和限制也可以选择性 merge 或者覆盖
                    if new_extracted.tags:
                        st.session_state.current_constraints.tags = list(set(st.session_state.current_constraints.tags + new_extracted.tags))
                except Exception as e:
                    # 即使提取失败，也可以选择降级忽略，只作为普通 restrictions 追加
                    pass

            # 3. 网关：获取知识
            c = st.session_state.current_constraints
            pois = mock_search_places(c.destination)
            weather = mock_get_weather(c.destination)
            budget_baseline = mock_estimate_budget_rules(c.destination)

            
            # 带相似度阈值过滤的合体检索， Chroma 默认返回的是距离，越小越相似
            # 用 vector_store 进行原生带分数检索，手动控分
            vector_store = st.session_state.global_retriever.vectorstore
            # 捞出最接近的 2 条数据和它们的分数
            # rag_search_query = f"{c.destination} {user_input}"
            search_keywords = [c.destination]
            if c.tags:
                search_keywords.extend(c.tags)
            
            rag_search_query = " ".join(search_keywords)
            docs_with_scores= vector_store.similarity_search_with_score(rag_search_query,k=2)
            # 设定距离阈值（Chroma 的 L2 距离，一般小于 0.6 或 0.7 说明语义非常相关）
            # 如果距离太大（比如 1.2），说明用户问的内容和知识库毫不相干
            valid_docs=[]
            for doc,score in docs_with_scores:
                if score: # 工业界调优参数，根据 Embedding 模型动态微调
                    valid_docs.append(doc)

            if valid_docs:
                rag_context = format_docs(valid_docs)
                st.caption(f"[RAG 命中]【匹配分数：{score}】 成功匹配到本地私有小红书知识。")
            else:
                rag_context = f"（注：本地知识库中未检索到与【{c.destination}】相关的私有攻略，请管家完全依赖自身的预训练知识或天气API为用户规划目的地行程，切勿混淆目的地。）"
                st.caption("[RAG 未命中] 未在本地库找到相关知识，已启用大模型通用知识规划。")

            # 动态执行rag检索
            # 将用户的输入扔进本地 ChromaDB 捞取最靠谱的避坑干货
            # retrieved_docs = retriever.invoke(user_input)
            # rag_context = format_docs(retrieved_docs)


            # 4. 驱动：规划（将 RAG 知识动态注入 restrictions 或作为独立上下文）
            history_context = (
                f"历史已生成方案参考：{st.session_state.current_plan.model_dump_json()}" 
                if st.session_state.current_plan else "这是首次规划"
            )
            # 融合 RAG 后的最终硬性限制条件
            final_restrictions = f"{c.restrictions} | 本地知识库避坑参考：{rag_context} | 历史上下文：{history_context}"

            try:
                final_plan: FullTravelPlan = planner_chain.invoke({
                    "destination": c.destination,
                    "duration_days": c.duration_days,
                    "budget": c.budget or "未指定",
                    "tags": c.tags,
                    "restrictions": final_restrictions,
                    "poi_data": str(pois),
                    "weather_data": str(weather),
                    "budget_baseline": budget_baseline,
                    "format_instructions": planner_parser.get_format_instructions()
                })
                st.session_state.current_plan = final_plan
            except Exception as e:
                st.error(f"规划链路解析失败：{e}")
                st.stop()

            # 5. 网页前端完美渲染渲染
            response_text = f"✨ 方案已为您更新！\n\n**💡 规划思路**: {final_plan.summary}"
            # 创建一个文本生成器，用于喂给 st.write_stream
            def text_generator():
                import time
                for word in response_text:
                    yield word
                    time.sleep(0.01) # 控制打字机速度
            # 使用 Streamlit 官方原生打字机组件
            st.write_stream(text_generator())
            # st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # 强制刷新页面以便右侧渲染最新的精美卡片看板
            st.rerun()

 
    # ================= 终极行程看板（若有数据则直接挂在聊天流下方） =================
if st.session_state.current_plan and st.session_state.current_constraints:
    plan: FullTravelPlan = st.session_state.current_plan
    c = st.session_state.current_constraints
    
    st.markdown("---")
    st.header(f"【{c.destination}】定制行程大盘看板")
    
    # ================= 1. 天气实时看板展示 =================
    st.subheader(" 当地实时天气看板")
    
    # 重新获取当前城市的天气数据（也可以在前面存入 session_state）
    current_weather = mock_get_weather(c.destination)
    
    # 用 Streamlit 的列布局把天气指标排开
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        st.metric(label=" 天气状况", value=current_weather.get("forecast", "未知"))
    with w_col2:
        st.metric(label=" 当前气温", value=current_weather.get("temperature", "未知"))
    with w_col3:
        # 根据是否有雨，展示不同的状态标签
        if "雨" in current_weather.get("forecast", ""):
            st.error(" 出行提示：有降雨预期")
        else:
            st.success(" 出行提示：天气适宜出行")
            
    # 联动大模型的智能风控预警
    if plan.weather_warning.has_risk:
        st.warning(f"**AI 智能风控提示**: {plan.weather_warning.tips}")
    
    st.markdown("---")
    # =======================================================
    # 2. 行程卡片流
    st.subheader(" 每日精细行程路线")
    for day in plan.daily_plans:
        with st.expander(f" 第 {day.day_number} 天具体行程（点击展开/折叠）", expanded=True):
            cols = st.columns(len(day.routes) if day.routes else 1)
            for idx, route in enumerate(day.routes):
                with cols[idx]:
                    st.markdown(f"** {route.time_slot}**")
                    st.info(f"📍 {route.location_name}")
                    st.markdown(f"**活动**: {route.activity}")
                    st.markdown(f"**交通**: {route.transport}")
                    st.caption(f" *决策逻辑*: {route.reason}")

    # 3. 账单图表化拆分
    st.subheader(" 预估精细账单拆分")
    
    # 组装表格数据
    bd = plan.budget_breakdown
    budget_data = {
        "开销项目": ["交通交通", "住宿开销", "餐饮美食", "景点门票", " 预估总计"],
        "预估金额 (元)": [bd.traffic, bd.accommodation, bd.food, bd.tickets, bd.total_estimated]
    }
    st.table(budget_data)
    
    if bd.is_over_budget:
        st.error(" **警告**：经智能风控系统测算，此方案开销可能会**超出**您的预期预算，请酌情删减住宿或餐饮档次！")
    else:
        st.success(" 该方案总预算完美控制在预期范围内。")