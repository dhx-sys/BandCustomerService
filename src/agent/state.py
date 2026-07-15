from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
#派单信息
class OrdersInfo(BaseModel):
    model_config = ConfigDict(strict=False)
    id: str= Field(description="id")
    thread_id: str = Field(description="线程id")
    user_id: str = Field(description="用户id")
    band_id: str = Field(description="宽带id")
    band_info: str = Field(description="宽带信息")
    band_address: str = Field(description="宽带地址")
    band_phone: str = Field(description="宽带套餐电话号码")
    user_phone: str = Field(description="联系电话")
    fault_type: str = Field(description="用户意图")
    fault_code: str = Field(description="故障编码")

class MessagesState(TypedDict):
    # 定义messages字段，类型为消息序列使用add_messages处理追加，
    messages: Annotated[Sequence[BaseMessage], add_messages]
    fault_type: str                  #用户意图
    ususer_score: bool             #是否本网用户
    intention_code: str
    say_to_user: str
    address: str    #匹配成功的宽带地址
    reason: str
    need_id_card_get_info: bool   #是否需要身份证号码查询宽带信息
    is_same_package_user_count: int #是否同套餐循环控制
    intent_recognize_count: int  # 是否解决宽带问题循环控制
    get_band_info_by_idcard_count: int  # 身份证号码查询控制循环控制
    matched_band_address_count: int  #地址匹配循环控制
    get_band_info: bool   #是否获取到宽带信息
    band_info: str  #接口查询的宽带信息
    fault_code: str #故障编码
    orders_info: OrdersInfo | None  #派单信息

class IntentRecognizeResult(BaseModel):
    model_config = ConfigDict(strict=False)
    fault_type: str = Field(description="用户意图")
    say_to_user: str = Field(description="下一轮给用户的回复")
    reason: str = Field(description="判断理由，用于调试")

class IsSamePackageResult(BaseModel):
    model_config = ConfigDict(strict=False)
    same_package: str = Field(default="nuknown", description="模型返回的结果:yes/no/nuknown")
    say_to_user: str = Field(description="下一轮给用户的回复")
    reason: str = Field(description="判断理由，用于调试")

class checkIdCardResult(BaseModel):
    model_config = ConfigDict(strict=False)
    check_result: str = Field(default="nuknown", description="模型返回的结果:yes/no")
    say_to_user: str = Field(description="下一轮给用户的回复")
    reason: str = Field(description="判断理由，用于调试")

class MatchedBandAddresSResult(BaseModel):
    model_config = ConfigDict(strict=False)
    check_address_result: str = Field(default="", description="模型返回的结果:yes/no")
    address: str = Field(default="", description="匹配到的地址")
    say_to_user: str = Field(default="", description="下一轮给用户的回复")
    reason: str = Field(default="", description="判断理由，用于调试")

class getFaultFodeResult(BaseModel):
    model_config = ConfigDict(strict=False)
    fault_ticket: bool= Field(description="是否需要派故障单处理")
    urge_ticket: bool = Field(description="是否需要催单处理")
    band_fault: bool = Field(description="网络是否已恢复")
    say_to_user: str = Field(description="下一轮给用户的回复")
    reason: str = Field(description="判断理由，用于调试")

