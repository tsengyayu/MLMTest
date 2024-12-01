# from openai import OpenAI
#
#
# # 查詢可用模型列表
# models = client.models.list()
# for model in models.data:
#     print(model.id)
#
#
# response = client.chat.completions.create(model="gpt-4o-mini",
# messages=[
#     {"role": "system", "content": "請用中文回答所有問題，並以專業且簡潔的語氣回答。"},
#     {"role": "user", "content": "你能告訴我如何使用Python執行這個指令嗎？"}
# ])
#
# # 顯示回應
# print(response.choices[0].message.content)

# import ollama
# import pandas as pd
# import pyreadr as pyr
# from tqdm.notebook import tqdm
# import os
#
# modelfile = """
# FROM llama3
# PARAMETER temperature 0.2
# PARAMETER num_ctx 2048
# SYSTEM 你是資料處裡的工具，你只會用英文回覆資料內容，不提供其他資訊
# """
#
# # create custom model
# # model參數可以自行替模型命名，此處命名為'data_processor'
# ollama.create(model='data_processor', modelfile=modelfile)
# ollama.list()
#
# # 啟用tqdm，讓pd.apply顯示進度條
# tqdm.pandas()
# #
# # # model參數填入剛剛建立的自訂模型
# # # prompt填入要送入模型的指令與資料
# # # 拉取模型
# # ollama.pull("data_processor")
# prompt = (f"Here are some examples of how to list synonyms:\n"
#               f"Word: teacher\n1. instructor\n2. educator\n3. mentor\n4. tutor\n5. professor\n"
#               f"Now, list 5 synonyms for the word 'student' in the sentence: 'i'm a student'."
#               f"Only output the synonyms in the format:\n1. synonym 1\n2. synonym 2\n3. synonym 3\n4. synonym 4\n5. synonym 5")
#
# data = ollama.generate(model='data_processor', prompt=prompt)
# name = data['response']
# print(name)

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
#
# def get_label(sentence):
#     inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     predictions = torch.softmax(outputs.logits, dim=-1)
#     predicted_class = torch.argmax(predictions, dim=-1).item()
#     return predicted_class
#
# #已使用yelp dataset的bert模型
# tokenizer = AutoTokenizer.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")
# model = AutoModelForSequenceClassification.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")
# new_label = get_label("2.")
# print(new_label)
#
# import shap
#
# # 定義一個封裝模型預測的函數
# def model_predict(texts):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     return torch.softmax(outputs.logits, dim=-1).detach().numpy()
# # 假設我們已經有一個訓練好的文本分類模型 'model' 和對應的 tokenizer
# explainer = shap.Explainer(model_predict, tokenizer)
#
# # 確保 text_instance 是單個字符串
# text_instance = "The movie was great!"  # 這是一個單個句子
#
# # 進行解釋時，確保傳遞的是一個包含字符串的列表
# shap_values = explainer([text_instance])
#
# # 可視化結果
# shap.plots.text(shap_values)







# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

# 使用已訓練好的 Yelp dataset 的 distilbert 模型
tokenizer = AutoTokenizer.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")
model = AutoModelForSequenceClassification.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")

# 將模型設置為評估模式
model.eval()

# 定義封裝模型預測的函數，用於 LIME
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1).detach().numpy()
    return probabilities

# 創建 LIME 解釋器
explainer = LimeTextExplainer(class_names=["negative", "positive"])

# 定義需要解釋的文本
text_instance = "Been going to Dr. Goldberg for over 10 years. I think I was one of his 1st patients when he started at MHMG. He's been great over the years and is really all about the big picture."

# 使用 LIME 解釋模型預測
exp = explainer.explain_instance(text_instance, predict_proba, num_features=10)
# 將關鍵詞及其影響以清單形式輸出
explanation = exp.as_list()
print(explanation)

html = exp.as_html()  # 生成 HTML 結果
with open('lime_explanation.html', 'w') as f:
    f.write(html)


# 顯示解釋結果
exp.show_in_notebook(text=True)





