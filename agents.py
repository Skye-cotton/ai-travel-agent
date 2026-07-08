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
    temperature=0.3,
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

planer_prompt = ChatPromptTemplate.from_messages([
    ('system', (
        "你是一个专业、严谨且注重风控的旅行规划师。\n"
        "你需要结合用户约束、POI地点、【实时天气】和【价格基准】来写行程。\n"
        "【硬性风控规则】:\n"
        "1. 如果天气预报有雨，必须在行程中提及备伞，或将室外活动尽量调至室内/乘船，并在 'reason' 中解释。\n"
        "2. 计算总预算区间，如果明显超过用户预算，必须在 'is_over_budget' 标记为 true，并在 summary 中主动致歉和解释方案削减。\n"
        "{format_instructions}"
    )),
    ("user", (
        "【用户约束】\n目的地: {destination} | 天数: {duration_days}天 | 预算上限: {budget}元\n"
        "偏好: {tags} | 限制: {restrictions}\n\n"
        "【外部实时信息】\n天气预报: {weather_data}\n价格参考基准: {budget_baseline}\n\n"
        "【备选地点 POI】\n{poi_data}\n\n"
        "请生成包含预算拆分和天气预警的完美结构化方案。"
    ))
]).partial(format_instructions=planner_parser.get_format_instructions())

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