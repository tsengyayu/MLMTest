import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")
model = AutoModelForSequenceClassification.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")

def get_label(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()
    return predicted_class



def replace_words_with_blank_and_check_label(sentence, original_label):
    # 將句子分割成詞語和標點符號
    words_with_punctuation = re.findall(r'\w+|[^\w\s]', sentence)

    # 複製一份用於替換的詞語列表
    modified_words = words_with_punctuation[:]

    # 逐詞替換成空白，並檢查每次替換的影響
    for i, token in enumerate(words_with_punctuation):
        # 將第 i 個詞替換成空白，並保持之前已替換的空白
        modified_words[i] = " "  # 將當前詞替換成空白
        temp_sentence = ' '.join(modified_words)  # 合成新的句子

        # 獲取替換後句子的標籤
        new_label = get_label(temp_sentence)

        # 印出結果
        print(f"Original Sentence: {sentence}")
        print(f"Modified Sentence with '{token}' replaced by blank: {temp_sentence}")
        print(f"Original Label: {original_label}, New Label: {new_label}")

        # 檢查標籤是否有變化
        if original_label == new_label:
            print("Label matches: True")
        else:
            print("Label matches: False")
        print("-" * 50)  # 分隔線

# 使用範例
original_sentence = "Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- good doctor, terrible staff.  It seems that his staff simply never answers the phone.  It usually takes 2 hours of repeated calling to get an answer.  Who has time for that or wants to deal with it?  I have run into this problem with many other doctors and I just don't get it.  You have office workers, you have patients with medical needs, why isn't anyone answering the phone?  It's incomprehensible and not work the aggravation.  It's with regret that I feel that I have to give Dr. Goldberg 2 stars."

# 逐詞替換並檢查標籤
replace_words_with_blank_and_check_label(original_sentence, "0")
