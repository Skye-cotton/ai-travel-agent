# 文件不包含任何 LLM 逻辑，只用来定义数据模型 这样可以防止循环导入（Circular Import）的问题。
from typing import List,Optional
from pydantic import BaseModel, Field

# 偏好提取模型
class TravelConstraints(BaseModel):
    destination: Optional[str] = Field(None, description="目的地城市，例如：杭州、成都。未提及则为 None。")
    duration_days: Optional[int] = Field(None, description="旅行天数，必须是数字。未提及则为 None。")
    budget: Optional[float] = Field(None, description="总预算（元），必须是数字。未提及则为 None。")
    tags: List[str] = Field(default=[], description="兴趣标签，如：风景、人文、美食、古镇、购物")
    restrictions: List[str] = Field(default=[], description="硬性限制或特殊需求，如：老人同行、不能走太多路、无障碍、不吃辣")

# 路线规划模型
class RouteItem(BaseModel):
    time_slot:str = Field(description="时间段，例如：上午 9:00-11:00")
    location_name:str = Field(description='景点或餐厅名称')
    activity:str = Field(description="在这个地方的具体名称")
    transport:str = Field(description="到这个地方的交通方式建议")
    reason:str = Field(description="为什么要这么安排（合理性可解释性）")
    # 强迫大模型输出可解释性理由，并标注不确定性风险
    why_this_arrangement: str = Field(description="为什么要这样安排？（结合用户的偏好或预算说明取舍逻辑）")
    source_grounding: str = Field(description="此信息的来源与不确定性披露（如：参考本地小红书库/参考大模型预训练知识，价格和营业时间存在变动风险）")
class DailyItinerary(BaseModel):
    day_number: int = Field(description="第几天，例如：1，2")
    routes: List[RouteItem] = Field(description='当天的行程打卡点列表')

# 添加预算与天气
class BudgetBreakdown(BaseModel):
    traffic: str = Field(description="交通预估开销，如：'100-200元（市内打车/地铁）'")
    accommodation: str = Field(description="住宿预估开销")
    food: str = Field(description="餐饮预估开销")
    tickets: str = Field(description="门票预估开销")
    total_estimated: str = Field(description="总计预估区间，例如：'1200-1500元'")
    is_over_budget: bool = Field(description="是否超过了用户给出的预算限制")
class SafetyHandoff(BaseModel):
    requires_human_confirmation: bool = Field(
        description="安全合规拦截闸门。如果用户提出了涉及金钱交易、直接扣款、要求代付、提供信用卡号密码、或是涉及人身安全的极端高风险动作，必须强制返回 True，否则返回 False。"
    )
    warning_message: str = Field(description="向用户揭示的高风险警告语，如涉及预订必须提示人工点击跳转")

class FullTravelPlan(BaseModel):
    summary: str = Field(description="整个行程的规划思路摘要")
    daily_plans: List[DailyItinerary] = Field(description="每日详细行程")
    budget_breakdown: BudgetBreakdown = Field(description="预算拆分精细账单")
    # 备选方案与安全机制
    alternatives_for_weather: str = Field(description="针对天气变化（如突然下雨）或突发情况的室内备选景点与路线预案")
    safety_review: SafetyHandoff = Field(description="Human-in-the-loop 安全合规审查结果")
