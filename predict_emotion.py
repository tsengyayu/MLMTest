from transformers import BertTokenizer,AutoTokenizer, AutoModelForSequenceClassification
import polars as pl
import pandas as pd
# 載入模型和 tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./kaggle/working/results/checkpoint-500')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

splits = {'train': 'data/train.jsonl', 'validation': 'data/validation.jsonl', 'test': 'data/test.jsonl'}
df_train = pl.read_ndjson('hf://datasets/AdamCodd/emotion-balanced/' + splits['train'])
df_train = df_train.to_pandas()
df_test = pl.read_ndjson('hf://datasets/AdamCodd/emotion-balanced/' + splits['test'])
df_test = df_test.to_pandas()

test_texts = df_test.head(1)['text'].tolist()
# 將測試句子進行 tokenization
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")

# 將測試數據傳入模型，進行預測
outputs = model(**test_encodings)

# 取得預測的標籤 (logits 是模型的輸出)
logits = outputs.logits
predictions = logits.argmax(dim=-1)

# 轉換為可讀的標籤
print("Predicted labels:", predictions.item())
