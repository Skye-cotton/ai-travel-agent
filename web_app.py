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
from langchain_chroma import Chroma
from langchain_zhipu import ZhipuAIEmbeddings
from rag_chain import format_docs

# 初始化本地RAG检索器
def init_local_retriever():
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 初始化 Session 状态
if "messages" not in st.session_state:
    st.session_state.messages = []  

if "current_constraints" not in st.session_state:
    st.session_state.current_constraints = None  

if "current_plan" not in st.session_state:
    st.session_state.current_plan = None  

# ================= 页面左侧栏：状态监视器 =================
with st.sidebar:
    st.title("记忆状态监视器")
    st.write("大模型当前捕获的结构化背景：")
    
    if st.session_state.current_constraints:
        c = st.session_state.current_constraints
        st.success(f"目的地: {c.destination or '未提取'}")
        st.info(f"天数: {c.duration_days or '未提取'} 天")
        st.warning(f"预算上限: {c.budget or '未指定'} 元")
        st.write(f"偏好标签: {', '.join(c.tags) if c.tags else '无'}")
        st.write(f"出行限制: {', '.join(c.restrictions) if c.restrictions else '无'}")
    else:
        st.caption("暂无结构化记忆，请在右侧输入您的旅行想法。")
        
    st.markdown("---")
    if st.button("清空对话记忆", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_constraints = None
        st.session_state.current_plan = None
        st.rerun()

# ================= 页面右侧主区：交互流 =================
st.title("您的全能智能旅行管家")
st.caption("基于 LangChain + Pydantic 强约束架构 | 实时注入天气与价格风控网关")

# 渲染历史聊天气泡
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 接收用户新输入
if user_input := st.chat_input("您可以说：我想带父母去杭州玩2天，别太累..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 状态驱动核心逻辑
    with st.chat_message("assistant"):
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
                st.stop()
        
        # 场景 B：已有行程，用户提修改意见
        else:
            st.session_state.current_constraints.restrictions.append(f"用户最新修改要求：{user_input}")
            try:
                chat_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                new_extracted: TravelConstraints = extractor_chain.invoke({
                    "user_input": f"历史对话上下文：\n{chat_context}\n用户最新输入：{user_input}",
                    "format_instructions": extractor_parser.get_format_instructions()
                })
                if new_extracted.destination:
                    st.session_state.current_constraints.destination = new_extracted.destination
                if new_extracted.duration_days:
                    st.session_state.current_constraints.duration_days = new_extracted.duration_days
                if new_extracted.budget:
                    st.session_state.current_constraints.budget = new_extracted.budget
                if new_extracted.tags:
                    st.session_state.current_constraints.tags = list(set(st.session_state.current_constraints.tags + new_extracted.tags))
            except Exception as e:
                pass

            # 再次校验约束完整性
            c = st.session_state.current_constraints
            if not c.destination or not c.duration_days:
                missing = []
                if not c.destination: missing.append("想去哪个城市")
                if not c.duration_days: missing.append("计划玩几天")
                response = f"信息还不够完整，请问您{' 和 '.join(missing)}呢？"
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

        # 3. 网关：获取知识
        c = st.session_state.current_constraints
        pois = mock_search_places(c.destination)
        weather = mock_get_weather(c.destination)
        budget_baseline = mock_estimate_budget_rules(c.destination)

        # 向量知识库检索
        # 带相似度阈值过滤的合体检索， Chroma 默认返回的是距离，越小越相似
        # 用 vector_store 进行原生带分数检索，手动控分
        vector_store = st.session_state.global_retriever.vectorstore
        search_keywords = [c.destination]
        if c.tags:
            search_keywords.extend(c.tags)
        
        rag_search_query = " ".join(search_keywords)
        docs_with_scores = vector_store.similarity_search_with_score(rag_search_query, k=2)
        
        valid_docs = []
        score = 0.0
        for doc, score in docs_with_scores:
            if score: # 工业界调优参数，根据 Embedding 模型动态微调
                valid_docs.append(doc)

        if valid_docs:
            rag_context = format_docs(valid_docs)
            st.caption(f"[RAG 命中][匹配分数：{score}] 成功匹配到本地私有知识。")
        else:
            rag_context = f"(注：本地知识库中未检索到与【{c.destination}】相关的私有攻略，请完全依赖自身的预训练知识或天气API进行规划。)"
            st.caption("[RAG 未命中] 未在本地库找到相关知识，已启用大模型通用知识规划。")

        # 4. 驱动：双轨流式规划生成与底层硬规则风控网关
        history_context = (
            f"历史已生成方案参考：{st.session_state.current_plan.model_dump_json()}" 
            if st.session_state.current_plan else "这是首次规划"
        )
        final_restrictions = f"{c.restrictions} | 本地知识库避坑参考：{rag_context} | 历史上下文：{history_context}"

        try:
            # 轨道一：聊天打字机。使用没有绑定 parser 的底层 llm 进行纯文本流式渲染，让用户免于等待
            # 这里直接通过原始的 planner_chain 调用它的底层文本流，或依赖模型的原生流
            # 如果你的组件不支持剥离，可以直接在前端提供统一的打字反馈
            def text_loading_generator():
                yield "正在为您检索实时天气数据...\n"
                yield "正在结合本地私有避坑知识库进行多维度偏好匹配...\n"
                yield "正在为您量身定制专属路线规划与预算风控审计..."
                
            st.write_stream(text_loading_generator())

            # 轨道二：后台全量强类型闭合解析。直接使用全量 invoke，确保 Pydantic 约束完美落地
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

            #  混合安全风控网关校验
            if st.session_state.current_plan:
                chat_history_str = "".join([m["content"] for m in st.session_state.messages]) + user_input
                high_risk_keywords = ["卡号", "信用卡", "帮我订", "帮我付款", "密码", "身份证", "护照"]
                
                if any(kw in chat_history_str for kw in high_risk_keywords):
                    st.session_state.current_plan.safety_review.requires_human_confirmation = True
                    st.session_state.current_plan.safety_review.warning_message = (
                        "系统检测到敏感身份凭证或越权预订指令。为了您的资金与隐私安全，"
                        "AI 已自动实施高风险阻断，请点击下方官方渠道手动完成预订。"
                    )

            if st.session_state.current_plan:
                response_text = f"方案已为您更新！\n\n**规划思路**: {st.session_state.current_plan.summary}"
                st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"规划链路解析失败：{e}")
            st.stop()

        st.rerun()

# ================= 终极行程看板（数据独立渲染区） =================
if st.session_state.current_plan and st.session_state.current_constraints:
    plan: FullTravelPlan = st.session_state.current_plan
    c = st.session_state.current_constraints
    
    st.markdown("---")
    st.header(f"【{c.destination}】定制行程大盘看板")
    
    st.subheader("当地实时天气看板")
    current_weather = mock_get_weather(c.destination)
    
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        st.metric(label="天气状况", value=current_weather.get("forecast", "未知"))
    with w_col2:
        st.metric(label="当前气温", value=current_weather.get("temperature", "未知"))
    with w_col3:
        if "雨" in current_weather.get("forecast", ""):
            st.error("出行提示：有降雨预期")
        else:
            st.success("出行提示：天气适宜出行")
            
    if plan.safety_review.requires_human_confirmation:
        st.error(f"Human-in-the-loop 安全风控提示: {plan.safety_review.warning_message}")
    else:
        if plan.alternatives_for_weather:
            st.info(f"自适应天气备选预案: {plan.alternatives_for_weather}")
    
    st.markdown("---")
    
    st.subheader("每日精细行程路线")
    for day in plan.daily_plans:
        with st.expander(f"第 {day.day_number} 天具体行程（点击展开/折叠）", expanded=True):
            cols = st.columns(len(day.routes) if day.routes else 1)
            for idx, route in enumerate(day.routes):
                with cols[idx]:
                    st.markdown(f"**{route.time_slot}**")
                    st.info(f"📍 {route.location_name}")
                    st.markdown(f"**活动**: {route.activity}")
                    st.markdown(f"**交通**: {route.transport}")
                    st.caption(f"*决策逻辑*: {route.reason}")

    st.subheader("预估精细账单拆分")
    bd = plan.budget_breakdown
    budget_data = {
        "开销项目": ["交通开销", "住宿开销", "餐饮美食", "景点门票", "预估总计"],
        "预估金额 (元)": [bd.traffic, bd.accommodation, bd.food, bd.tickets, bd.total_estimated]
    }
    st.table(budget_data)
    
    if bd.is_over_budget:
        st.error("警告：经智能风控系统测算，此方案开销可能会超出您的预期预算，请酌情删减住宿或餐饮档次！")
    else:
        st.success("该方案总预算完美控制在预期范围内。")