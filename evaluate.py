import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float32)

text = "what's your name"

encoded_input = tokenizer(text, return_tensors='pt').to(device)
model.to(device)

output = model(**encoded_input)

# 解碼輸入的 Token ID 回原始文本
decoded_text = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)
print("Decoded Input Text:", decoded_text)

# 顯示 Token ID 和對應的 Tokens
tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
print("Tokens:", tokens)

# 如果需要，可以打印嵌入對應的每個 Token
hidden_states = output.last_hidden_state[0]  # 第一個輸入的句子嵌入
for token, embedding in zip(tokens, hidden_states):
    print(f"Token: {token}, Embedding (first 5 values): {embedding[:5].tolist()}")