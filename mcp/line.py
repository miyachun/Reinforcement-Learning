"""
FastMCP + LINE Bot 整合範例
使用 FastMCP 處理 AI 功能，透過 LINE Bot 提供服務
"""

from fastapi import FastAPI, Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    QuickReply,
    QuickReplyItem,
    MessageAction
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from mcp.server.fastmcp import FastMCP
import os
from typing import Optional
import json


mcp = FastMCP("LINE Bot MCP Server")

@mcp.tool()
def translate_text(text: str, target_language: str = "en") -> str:
    """
    翻譯文字
    
    Args:
        text: 要翻譯的文字
        target_language: 目標語言 (en/zh/ja/ko)
    """
    # 這裡可以接入真實的翻譯 API
    translations = {
        "en": f"[EN] {text}",
        "zh": f"[中文] {text}",
        "ja": f"[日本語] {text}",
        "ko": f"[한국어] {text}"
    }
    return translations.get(target_language, f"[{target_language}] {text}")


@mcp.tool()
def analyze_sentiment(text: str) -> dict:
    """
    分析文字情緒
    
    Args:
        text: 要分析的文字
    """
    
    positive_words = ["開心", "快樂", "喜歡", "好", "讚", "棒", "love", "happy", "good"]
    negative_words = ["難過", "討厭", "壞", "爛", "糟", "sad", "hate", "bad"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "正面 😊"
        score = 0.7
    elif negative_count > positive_count:
        sentiment = "負面 😔"
        score = 0.3
    else:
        sentiment = "中性 😐"
        score = 0.5
    
    return {
        "sentiment": sentiment,
        "score": score,
        "positive_count": positive_count,
        "negative_count": negative_count
    }


@mcp.tool()
def get_weather_info(city: str) -> dict:
    """
    獲取天氣資訊
    
    Args:
        city: 城市名稱
    """
    
    weather_data = {
        "台北": {"temp": 25, "condition": "多雲", "humidity": 70},
        "台中": {"temp": 28, "condition": "晴天", "humidity": 65},
        "高雄": {"temp": 30, "condition": "晴天", "humidity": 75},
        "taipei": {"temp": 25, "condition": "Cloudy", "humidity": 70},
    }
    
    city_lower = city.lower()
    if city_lower in weather_data or city in weather_data:
        data = weather_data.get(city_lower) or weather_data.get(city)
        return {
            "city": city,
            "temperature": data["temp"],
            "condition": data["condition"],
            "humidity": data["humidity"]
        }
    
    return {
        "city": city,
        "error": "找不到該城市的天氣資訊"
    }


@mcp.tool()
def calculate(expression: str) -> str:
    """
    計算數學表達式
    
    Args:
        expression: 數學表達式（如 "2+3*4"）
    """
    try:
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"計算錯誤: {str(e)}"


@mcp.resource("config://bot")
def get_bot_config() -> str:
    """獲取機器人配置"""
    config = {
        "name": "MCP LINE Bot",
        "version": "1.0.0",
        "features": [
            "文字翻譯",
            "情緒分析",
            "天氣查詢",
            "數學計算"
        ]
    }
    return json.dumps(config, ensure_ascii=False, indent=2)


@mcp.prompt()
def chat_prompt(user_message: str) -> str:
    """生成聊天回覆的提示詞"""
    return f"""你是一個友善的 LINE Bot 助手。

用戶訊息: {user_message}

請用溫暖、友善的語氣回覆用戶。如果用戶需要幫助，請告訴他們可以使用以下功能：
- 翻譯文字（輸入：翻譯 [文字]）
- 情緒分析（輸入：分析 [文字]）
- 查詢天氣（輸入：天氣 [城市]）
- 數學計算（輸入：計算 [表達式]）
"""

app = FastAPI()

LINE_CHANNEL_ACCESS_TOKEN = 'XXXX'
LINE_CHANNEL_SECRET = 'XXXX'

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def process_message(text: str) -> str:    
    
    text = text.strip()   

    if text.startswith("翻譯"):
        content = text[2:].strip()
        if content:
            result = translate_text(content, "en")
            return f"翻譯結果:\n{result}"
        return "請輸入要翻譯的文字，格式：翻譯 [文字]"
    

    elif text.startswith("分析"):
        content = text[2:].strip()
        if content:
            result = analyze_sentiment(content)
            return f"""情緒分析結果:
情緒: {result['sentiment']}
分數: {result['score']:.2f}
正面詞彙: {result['positive_count']} 個
負面詞彙: {result['negative_count']} 個"""
        return "請輸入要分析的文字，格式：分析 [文字]"
    

    elif text.startswith("天氣"):
        city = text[2:].strip()
        if city:
            result = get_weather_info(city)
            if "error" in result:
                return result["error"]
            return f"""🌤 {result['city']} 天氣資訊:
溫度: {result['temperature']}°C
天氣: {result['condition']}
濕度: {result['humidity']}%"""
        return "請輸入城市名稱，格式：天氣 [城市]"
    

    elif text.startswith("計算"):
        expression = text[2:].strip()
        if expression:
            result = calculate(expression)
            return f"🔢 {result}"
        return "請輸入數學表達式，格式：計算 [表達式]"
    

    elif text in ["幫助", "help", "功能"]:
        return """🤖 我可以幫你做這些事:

📝 翻譯文字
格式: 翻譯 [文字]
範例: 翻譯 你好

💭 情緒分析
格式: 分析 [文字]
範例: 分析 今天天氣真好

🌤 查詢天氣
格式: 天氣 [城市]
範例: 天氣 台北

🔢 數學計算
格式: 計算 [表達式]
範例: 計算 2+3*4

輸入「幫助」查看此訊息"""
    
    
    else:
        return f"你說: {text}\n\n不確定你的意思？輸入「幫助」查看我能做什麼！"



@app.post("/webhook")
async def webhook(request: Request):           
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature")   
    
    body = await request.body()
    body_str = body.decode("utf-8")    
    
    try:
        handler.handle(body_str, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):    
    user_message = event.message.text
    reply_text = process_message(user_message)
    quick_reply = QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="翻譯", text="翻譯 ")),
        QuickReplyItem(action=MessageAction(label="分析", text="分析 ")),
        QuickReplyItem(action=MessageAction(label="天氣", text="天氣 台北")),
        QuickReplyItem(action=MessageAction(label="計算", text="計算 ")),
        QuickReplyItem(action=MessageAction(label="幫助", text="幫助")),
    ])
    

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text, quick_reply=quick_reply)]
            )
        )


if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=5000)
    #mcp.run()