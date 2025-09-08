import tkinter as tk
import torch
import random
import json
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from datetime import datetime
import csv
import requests
from PIL import Image, ImageTk
from io import BytesIO

# WeatherAPI Key（记得替换为你自己的）
WEATHER_API_KEY = "e997151541c24061b4d123258251107"

# 上下文记忆
context_memory = {}

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

data = torch.load("data.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# FAQ 文件检查
def check_faq(msg):
    try:
        with open('faq.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['question'].lower() in msg.lower():
                    return row['answer']
    except FileNotFoundError:
        return None
    return None

# 关键词分类
custom_keywords = {
    "delivery": "faq_shipping",
    "return": "faq_refund",
    "payment": "faq_payment",
    "order": "faq_order"
}

# 获取天气信息
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&lang=zh"
    try:
        response = requests.get(url)
        data = response.json()

        if "error" in data:
            return "❌ 城市找不到或 API 错误", None

        location = data["location"]["name"]
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        icon_url = "https:" + data["current"]["condition"]["icon"]
        return f"🌤️ {location} 当前 {temp_c}°C，{condition}", icon_url
    except:
        return "❌ 获取天气失败，请检查网络连接", None

# 显示天气图标
def show_weather_icon(icon_url):
    try:
        response = requests.get(icon_url)
        img_data = response.content
        img = Image.open(BytesIO(img_data))
        img = img.resize((64, 64))
        photo = ImageTk.PhotoImage(img)
        icon_label.config(image=photo)
        icon_label.image = photo
    except:
        icon_label.config(image='')

# 获取聊天回应
def get_response(msg, user_id="user"):
    context_memory[user_id] = msg
    msg = msg.lower()

    if msg.startswith("/weather"):
        parts = msg.split(maxsplit=1)
        if len(parts) < 2:
            return "❗ 格式错误，请使用 /weather <城市名>"
        city = parts[1]
        weather_info, icon_url = get_weather(city)
        if icon_url:
            show_weather_icon(icon_url)
        return weather_info

    if "time" in msg or "date" in msg:
        return "🕒 现在是 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    faq_answer = check_faq(msg)
    if faq_answer:
        return "📚 " + faq_answer

    for keyword in custom_keywords:
        if keyword in msg:
            return f"🔍 你在问关于 {keyword} 的问题，请访问我们的帮助页面！"

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])

    return "🤔 抱歉，我不太理解你的意思..."

# GUI 操作
def send():
    user_input = entry.get()
    if not user_input:
        return
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    response = get_response(user_input)
    chat_log.insert(tk.END, f"Bot: {response}\n\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

# 界面设置
window = tk.Tk()
window.title("Smart Chatbot")
window.geometry("500x600")

chat_log = tk.Text(window, bd=1, bg="white", font=("Arial", 12))
chat_log.config(state=tk.DISABLED)
chat_log.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

entry = tk.Entry(window, font=("Arial", 14))
entry.pack(fill=tk.X, padx=10, pady=5)
entry.bind("<Return>", lambda event=None: send())

send_button = tk.Button(window, text="Send", command=send)
send_button.pack(pady=5)

icon_label = tk.Label(window)
icon_label.pack(pady=5)

window.mainloop()
