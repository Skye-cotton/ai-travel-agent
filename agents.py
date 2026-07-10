from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_zhipu import ChatZhipuAI
from models import TravelConstraints, FullTravelPlan
import os
from dotenv import load_dotenv
import requests

# 新增导入：用于解析结构化输出
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv(override=True)

# 初始化模型组件
# 提取任务用低温度的 Flash，追求精准
extractor_llm = ChatZhipuAI(
    model='glm-4-flash',
    temperature=0.1,
    zhipuai_api_key=os.getenv('ZHIPUAI_API_KEY')
)

# 规划任务用逻辑更强的 Plus，给一点点创造力
planner_llm = ChatZhipuAI(
    model="glm-4-plus",
    temperature=0.01,
    zhipuai_api_key=os.getenv('ZHIPUAI_API_KEY')
)

# ========== 1. 偏好提取链（使用 PydanticOutputParser） ==========
# 创建解析器，指定输出模型为 TravelConstraints
extractor_parser = PydanticOutputParser(pydantic_object=TravelConstraints)

# 构建 prompt，在系统消息中插入格式说明（{format_instructions}）
preference_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一个严谨的旅行助手。你的唯一任务是从用户的模糊描述中精确提取出旅游要素。"
     "不要主观臆断或编造用户没有提供的信息。\n{format_instructions}"
    ),
    ("user", "用户输入:{user_input}")
]).partial(format_instructions=extractor_parser.get_format_instructions())

# 组装链：prompt → llm → parser（不再使用 with_structured_output）
extractor_chain = preference_prompt | extractor_llm | extractor_parser

# ========== 2. 行程规划链（同样使用 PydanticOutputParser） ==========
planner_parser = PydanticOutputParser(pydantic_object=FullTravelPlan)

planner_template = """你是一位持有国家高级导游证、同时具备极强合规风控意识的【智能规划 Agent 专家】。
请根据以下多源异构数据、历史记忆以及安全规范，为用户定制一套无懈可击的旅行规划方案。

【输入的多源异构数据资产】
1. 目的地: {destination}
2. 计划游玩天数: {duration_days} 天
3. 用户预算上限: {budget} 元
4. 用户的核心偏好标签: {tags}
5. 硬性限制条件 & 外部 RAG 避坑线索: {restrictions}
6. 当地候选 POI 数据池: {poi_data}
7. 当地实时天气网关数据: {weather_data}
8. 行业基础物价基准线: {budget_baseline}

【四大核心工作法则】
1. 【可解释性 (Interpretability)】：在每一项行程的 'why_this_arrangement' 中，清晰阐述你的取舍逻辑（例如：‘因用户带老人，故放弃爬山，改坐缆车’，或‘因预算紧张，放弃黑珍珠餐厅，选择小红书推荐的高性价比本地小吃’）。
2. 【数据扎根 (Grounding) 与不确定性暴露】：禁止编造绝对的价格或绝对不塞车的承诺。在 'source_grounding' 中必须清晰标注信息来源与潜在变动风险。
3. 【天气自适应弹性 (Weather Adaptability)】：必须阅读实时天气数据，如果是阴雨天，行程必须避开露天大草坪，并且在 'alternatives_for_weather' 里留下详细的室内备选场馆预案。
4. 【Human-in-the-loop 安全金钟罩（绝对死线）】：
仔细审查 'restrictions' 和用户输入。如果发现任何包含以下特征的请求：
- 显式提供了信用卡号、密码、身份证号、护照号等敏感凭证。
- 要求系统代为执行实质性付款、预订酒店、购买机票、确认订单、代付扣款。
你必须、且只能将 'safety_review' 中的 'requires_human_confirmation' 设为 true。并在 'warning_message' 中明确写道：“检测到高风险越权操作，系统已自动拦截。涉及资金与隐私安全，请点击下方链接跳转到官方平台手动完成。”。绝对禁止擅自通过或直接规划付款流程！

请严格按照以下格式要求进行结构化输出：
{format_instructions}"""

planer_prompt = ChatPromptTemplate.from_template(planner_template)

planner_chain = planer_prompt | planner_llm | planner_parser

# ========== 以下 mock 工具函数保持不变 ==========
def mock_search_places(city: str) -> List[dict]:
    database = {
        "杭州": [
            {"name": "西湖风景名胜区", "type": "风景", "desc": "地标，平坦适合散步，适合老人"},
            {"name": "灵隐寺", "type": "人文", "desc": "千年古刹，需要爬一点台阶"},
            {"name": "河坊街", "type": "美食", "desc": "历史小吃街，晚上很热闹，有正宗本帮菜"},
            {"name": "西溪国家湿地公园", "type": "风景", "desc": "乘船游览，非常轻松舒适"},
            {"name": "外婆家(湖滨店)", "type": "美食", "desc": "正宗杭帮菜，性价比高"}
        ]
    }
    return database.get(city, [{"name": f"{city}通用景点", "type": "通用", "desc": "基础推荐"}])

def mock_get_weather(city: str) -> dict:

    """接入天气 API """
    api_key = os.getenv('QWEATHER_API_KEY')
    if not api_key:
        # 如果没配置 Key，降级回 Mock 数据，防止报错
        return {"forecast": "未配置天气API，模拟第二天有中雨", "temperature": "18-23度"}
    try:
        # 模糊搜索城市的location ID
        geo_url = f"https://geoapi.qweather.com/v2/city/lookup?location={city}&key={api_key}"
        geo_res = requests.get(geo_url, timeout=5).json()

        if geo_res.get("code") != '200' or not geo_res.get("location"):
            return {"forecast": "未知天气", "temperature": "未知"}
        location_id = geo_res["location"][0]["id"]
        # 2. 查询该城市未来 3 天的天气预报
        weather_url = f"https://devapi.qweather.com/v7/weather/3d?location={location_id}&key={api_key}"
        weather_res = requests.get(weather_url, timeout=5).json()
        if weather_res.get("code") != '200' or not weather_res.get("daily"):
            return {"forecast": "天气获取失败", "temperature": "未知"}
        # 提取第二天（明天）天气
        tomorrow_data = weather_res["daily"][1]
        forecast_text = f"明天天气：{tomorrow_data['textDay']}转{tomorrow_data["textNight"]}"
        tem_text = f"{tomorrow_data['tempMin']}-{tomorrow_data['tempMax']}"
        return {"forecast": forecast_text, "temperature": tem_text}
    except Exception as e:
        return {"forecast": f"天气接口异常: {str(e)}", "temperature": "未知"}

def mock_estimate_budget_rules(city: str) -> str:
    """模拟一些静态的价格区间参考基准"""
    return "杭州酒店旺季均价 400-600元/晚，杭帮菜人均 80-150元/餐，西湖免费，灵隐寺门票+香火券约 75元。"