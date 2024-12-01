import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
import spacy
from nltk.corpus import stopwords
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.corpus import wordnet
import openai

# 載入 GPT-2 模型和 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
stop_words = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

# 將標點符號與詞語分開的函數
def spacy_tokenize(sentence):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]  # 使用 spaCy 自動分開詞語和標點符號
    return ' '.join(tokens)

# 分離標點符號後再使用的版本
def process_sentences(sentences):
    return [spacy_tokenize(sentence) for sentence in sentences]

splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["train"])

original_sentences = df.head(1)['text'].tolist()

# 使用標點符號處理的版本
processed_sentences = process_sentences(original_sentences)

### 生成重要詞語排序 ###
pos_masks = []  # 根據每個句子的長度生成 mask 長度
for sentence in processed_sentences:
    num_words_original = len(sentence.split())  # 使用處理過的句子
    pos_masks.append({
        'original': list(range(num_words_original)),
    })

original_importance_scores = []

# 載入預訓練的 BERT 模型
enc = BertTokenizer.from_pretrained('bert-base-uncased')
mlm_model_ts = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 生成掩碼句子
def mask_word_in_sentence(sentence, pos):
    words = sentence.split()
    if pos < len(words):  # 檢查是否超出範圍
        words[pos] = '[MASK]'
    return ' '.join(words)

# 計算詞語重要性
def get_word_importance(masked_sentences, original_sentences, pos_masks):
    encoded_inputs = enc(masked_sentences, return_tensors='pt', padding='max_length', max_length=128)  # 編碼句子
    outputs = mlm_model_ts(**encoded_inputs)

    original_probs = []
    for i, pos in enumerate(pos_masks):
        if pos < len(original_sentences[i].split()):  # 檢查是否超出範圍
            original_token = enc.tokenize(original_sentences[i].split()[pos])  # 原始詞 token
            original_token_id = enc.convert_tokens_to_ids(original_token[0])  # 取首個子詞的 token id
            original_prob = torch.softmax(outputs.logits[i, pos, :], dim=-1)[original_token_id].item()
            original_probs.append(original_prob)

    return original_probs

# 檢查詞語是否有效
def is_valid_word(word, pos_tag):
    return (word.lower() not in stop_words and
            pos_tag not in ['PRON', 'DET', 'PUNCT', 'SYM'] and  # 過濾標點和符號
            word.isalpha())  # 檢查詞語是否僅包含字母

# 計算詞語的重要性
for i, sentence in enumerate(processed_sentences):
    doc_original = nlp(sentence)  # 對處理過的句子進行詞性標註

    original_sentence_importance = []
    for pos in pos_masks[i]['original']:
        if pos < len(sentence.split()):  # 確保索引不超出範圍
            word = sentence.split()[pos]
            pos_tag = doc_original[pos].pos_  # 詞性標註
            if is_valid_word(word, pos_tag):  # 檢查是否應該替換
                masked_sentence = mask_word_in_sentence(sentence, pos)
                masked_sentences = [masked_sentence]
                original_prob = get_word_importance(masked_sentences, [sentence], [pos])[0]
                original_sentence_importance.append((word, original_prob))

    original_sentence_importance.sort(key=lambda x: x[1])
    original_importance_scores.append(original_sentence_importance)

# 打印結果
for i, original_importance in enumerate(original_importance_scores):
    print(f"Sentence {i + 1} original word importance ranking:")
    for word, score in original_importance:
        print(f"{word}: {score}")
    print()


def synonym_replacement(original_sentence, word):
    return get_synonym_from_gpt2(original_sentence, word)  # 使用 LLM 生成同義詞



# 計算前 30% 的詞語
def replace_top_30_percent_words(original_sentence, importance_scores):
    num_words_to_replace = max(1, int(0.3 * len(importance_scores)))  # 計算要替換的詞數量
    words_to_replace = [word for word, score in importance_scores[:num_words_to_replace]]  # 取前 30% 的詞語
    replaced_sentence = []

    # 替換詞語
    for word in original_sentence.split():
        if word in words_to_replace:
            replaced_sentence.append(synonym_replacement(original_sentence, word))  # 用同義詞替換
        else:
            replaced_sentence.append(word)

    return ' '.join(replaced_sentence)

# def get_synonym_from_llm(word):
#     prompt = f"Generate a synonym for the word '{word}' that can confuse a text classification model."
#
#     try:
#         response = openai.Completion.create(
#             engine="text-davinci-003",  # 或其他可用模型
#             prompt=prompt,
#             max_tokens=10,
#             n=1,
#             stop=None,
#             temperature=0.7
#         )
#
#         synonym = response.choices[0].text.strip()
#         return synonym if synonym else word  # 如果生成無效，返回原詞
#     except Exception as e:
#         print(f"Error generating synonym for {word}: {e}")
#         return word  # 如果出現錯誤，返回原詞

# def get_synonym_from_llm(word):
#     prompt = f"Generate a synonym for the word '{word}' that can confuse a text classification model."
#
#     try:
#         response = openai.completions.create(
#             model="gpt-3.5-turbo-instruct",  # 適合新版本的模型名稱
#             prompt=prompt,
#             max_tokens=10,
#             n=1,
#             temperature=0.7
#         )
#
#         synonym = response.choices[0].text.strip()
#         return synonym if synonym else word  # 如果生成無效，返回原詞
#     except Exception as e:
#         print(f"Error generating synonym for {word}: {e}")
#         return word  # 如果出現錯誤，返回原詞

import re
# 生成同義詞的函數
def get_synonym_from_gpt2(current_text, word):
    # prompt = f"Generate a rare synonym or a less common related word for'{word}' that is still similar in meaning but could confuse a text classification model."
    # prompt = f"Given the following sentence:'{current_text} ', find a list of k synonyms for the word'{word}' within the sentence. " \
    #          f"When replacing the word with its synonyms, the new sentence should maintain the semantic meaning, syntax, and grammar " \
    #          f"of the original sentence. You should only output the synonyms of the word but not the sentences containing them. " \
    #          f"You should also format your output as following: '1. word 1\n 2. word 2\n 3. word 3\n ... 5. word 5\n'. " \
    #          f"Furthermore, you should rank your results by the semantic similarity between the original sentence and " \
    #          f"the sentence after replacing the word with its synonyms.";
    # prompt = (f"Find 5 synonyms for the word '{word}' in the sentence: '{current_text}'. "
    #           f"Only output the synonyms, formatted as follows:\n"
    #           f"1. synonym 1\n2. synonym 2\n3. synonym 3\n4. synonym 4\n5. synonym 5")
    prompt = (f"Here are some examples of how to list synonyms:\n"
              f"Word: teacher\n1. instructor\n2. educator\n3. mentor\n4. tutor\n5. professor\n"
              f"Now, list 5 synonyms for the word '{word}' in the sentence: '{current_text}'."
              f"Only output the synonyms in the format:\n1. synonym 1\n2. synonym 2\n3. synonym 3\n4. synonym 4\n5. synonym 5")

    # 將 prompt 編碼成 GPT-2 的輸入格式
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # 使用 GPT-2 生成文本
    outputs = model.generate(
        inputs,
        max_length=200,  # 設置生成文本的最大長度
        num_return_sequences=1,  # 返回一個生成的序列
        temperature=0.3,  # 控制生成文本的多樣性
        top_k=5,  # 限制 GPT-2 在每一步只從前 10 個最高機率的詞中選擇
        # top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # 確保正確設置
    )

    # 將輸出結果拆解成所需的同義詞列表
    print(outputs)
    generated_text = tokenizer.decode(outputs[0].tolist())
    print(generated_text)

    # 使用正則表達式來提取同義詞
    synonyms = re.findall(r'\d\.\s(\w+)', generated_text)
    print("synonyms: "+str(synonyms))
    print(synonyms)

    # 如果找不到同義詞，返回原單詞
    if not synonyms:
        return word

    return synonyms[0]  # 選擇第一個同義詞

# 測試同義詞生成
word = "student"
current_text = "I'm a student, and I like going to school.";
synonym = get_synonym_from_gpt2(current_text, word)
print(f"Synonym for '{word}': {synonym}")


# 計算詞語的重要性
for i, sentence in enumerate(original_sentences):
    doc_original = nlp(sentence)  # 對原始句子進行詞性標註

    original_sentence_importance = []
    for pos in pos_masks[i]['original']:
        if pos < len(sentence.split()):  # 檢查是否超出範圍
            word = sentence.split()[pos]
            pos_tag = doc_original[pos].pos_  # 詞性標註
            if is_valid_word(word, pos_tag):  # 檢查是否應該替換
                masked_sentence = mask_word_in_sentence(sentence, pos)
                masked_sentences = [masked_sentence]
                original_prob = get_word_importance(masked_sentences, [sentence], [pos])[0]
                original_sentence_importance.append((word, original_prob))

    # 對詞語的重要性排序
    original_sentence_importance.sort(key=lambda x: x[1])

    # 進行前 30% 的同義詞替換
    new_sentence = replace_top_30_percent_words(sentence, original_sentence_importance)
    print(f"Original Sentence {i + 1}: {sentence}")
    print(f"New Sentence {i + 1} with LLM Synonym Replacement: {new_sentence}")