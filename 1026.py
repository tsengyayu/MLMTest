import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import spacy
from nltk.corpus import stopwords
import re
import openai
import numpy as np
import tensorflow_hub as hub
# from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

openai.api_key = "your_openai_api_key"
nlp = spacy.load("en_core_web_sm") #加載NLP模型
stop_words = set(stopwords.words('english')) #停用詞

tokenizer = GPT2Tokenizer.from_pretrained("gpt2") #加載GPT-2模型
model = GPT2LMHeadModel.from_pretrained("gpt2")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #加載BERT模型用於MLM
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") #計算語意相似性

# model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb") #加載BERT模型用於evaluation
# tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
# model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

def spacy_tokenize(sentence): # 句子標記化處理，分割成一系列詞語和標點符號
    doc = nlp(sentence)
    return ' '.join([token.text for token in doc])

def mask_word_in_sentence(sentence, pos): #將詞語替換為mask，用於計算詞語重要性計算
    words = sentence.split()
    if pos < len(words):
        words[pos] = '[MASK]'
    return ' '.join(words)

def get_word_importance(masked_sentences, original_sentences, pos_masks): #對被遮罩的句子，使用MLM計算每個詞的重要性
    encoded_inputs = bert_tokenizer(masked_sentences, return_tensors='pt', padding='max_length', max_length=128)
    outputs = mlm_model(**encoded_inputs)

    original_probs = []
    for i, pos in enumerate(pos_masks):
        if pos < len(original_sentences[i].split()):
            original_token = bert_tokenizer.tokenize(original_sentences[i].split()[pos])
            original_token_id = bert_tokenizer.convert_tokens_to_ids(original_token[0])
            original_prob = torch.softmax(outputs.logits[i, pos, :], dim=-1)[original_token_id].item()
            original_probs.append(original_prob)

    return original_probs

def is_valid_word(word, pos_tag): # 檢查詞語是否為有效的替換對象
    # Ensure word is valid (not a stop word, proper noun, or invalid POS)
    return (word.lower() not in stop_words and
            pos_tag not in ['PRON', 'DET', 'PUNCT', 'SYM'] and word.isalpha())

def protect_named_entities(doc): # 找到句子中的命名實體和代詞，避免被替換掉
    protected_indices = set()
    for ent in doc.ents:
        for token in ent:
            protected_indices.add(token.i)
    return protected_indices

def calculate_use_similarity(original_sentence, modified_sentence): #計算原句和替換後句子之間的語意相似性，以保持語意一致性
    original_embedding = use_model([original_sentence])
    modified_embedding = use_model([modified_sentence])
    similarity = np.inner(original_embedding, modified_embedding)
    return similarity[0][0]

def get_synonym_from_gpt2(current_text, word):
    prompt = (f"Here are some examples of how to list synonyms:\n"
              f"Word: teacher\n1. instructor\n2. educator\n3. mentor\n4. tutor\n5. professor\n"
              f"Now, list 5 synonyms for the word '{word}' in the sentence: '{current_text}'."
              f"Only output the synonyms in the format:\n1. synonym 1\n2. synonym 2\n3. synonym 3\n4. synonym 4\n5. synonym 5")

    # 設置 pad_token，使用 eos_token 或新增填充標記
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )

    outputs = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=300, num_return_sequences=1,
        temperature=0.3, top_k=5, do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0].tolist())
    synonyms = re.findall(r'\d\.\s(\w+)', generated_text)

    return synonyms[0] if synonyms else word

def replace_top_30_percent_words(original_sentence, importance_scores, threshold=0.9): #選擇前30%的重要詞語進行同義詞替換
    importance_scores_sorted = sorted(importance_scores, key=lambda x: x[1], reverse=True)
    num_words_to_replace = max(1, int(0.3 * len(importance_scores_sorted)))
    words_to_replace = [word for word, score in importance_scores[:num_words_to_replace]]
    replaced_sentence = []

    words_with_punctuation = re.findall(r'\w+|[^\w\s]', original_sentence)

    for token in words_with_punctuation:
        word_without_punct = re.sub(r'[^\w]', '', token)
        if word_without_punct in words_to_replace:
            synonym = get_synonym_from_gpt2(original_sentence, word_without_punct)
            temp_sentence = original_sentence.replace(word_without_punct, synonym)

            # Use Universal Sentence Encoder to check semantic similarity
            if calculate_use_similarity(original_sentence, temp_sentence) >= threshold:
                replaced_token = token.replace(word_without_punct, synonym)
                replaced_sentence.append(replaced_token)
            else:
                replaced_sentence.append(token)
        else:
            replaced_sentence.append(token)

    return ' '.join(replaced_sentence)

#載入資料
splits = {'train': 'plain_text/train-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["train"])

#MLM詞語重要性排序
original_sentences = df.head(1)['text'].tolist() #取第一筆數據
processed_sentences = [spacy_tokenize(sentence) for sentence in original_sentences]
print("processed_sentences", processed_sentences)
pos_masks = [{'original': list(range(len(sentence.split())))} for sentence in processed_sentences]
original_importance_scores = []

for i, sentence in enumerate(processed_sentences):
    doc_original = nlp(sentence)
    protected_indices = protect_named_entities(doc_original)
    original_sentence_importance = []

    for pos in pos_masks[i]['original']:
        if pos not in protected_indices and pos < len(sentence.split()):
            word = sentence.split()[pos]
            pos_tag = doc_original[pos].pos_
            if is_valid_word(word, pos_tag):
                masked_sentence = mask_word_in_sentence(sentence, pos)
                original_prob = get_word_importance([masked_sentence], [sentence], [pos])[0]
                original_sentence_importance.append((word, original_prob))

    original_sentence_importance.sort(key=lambda x: x[1])
    original_importance_scores.append(original_sentence_importance)
    print("original_importance_scores", original_importance_scores)

#LLM生成同義詞
for i, original_sentence in enumerate(original_sentences):
    new_sentence = replace_top_30_percent_words(original_sentence, original_importance_scores[i])
    print(f"Original Sentence {i + 1}: {original_sentence}")
    print(f"New Sentence {i + 1} with LLM Synonym Replacement: {new_sentence}")

    # # Evaluate the new sentence using the TextAttack BERT model
    # inputs = tokenizer(new_sentence, return_tensors="pt", padding=True, truncation=True)
    # outputs = model_wrapper.model(**inputs)
    #
    # # Get prediction results
    # predictions = torch.softmax(outputs.logits, dim=-1)
    # predicted_class = torch.argmax(predictions, dim=-1).item()
    # probabilities = predictions[0].tolist()
    #
    # print(f"Predicted class for new sentence: {predicted_class}")
    # print(f"Class probabilities: {probabilities}")

#所有數據放入NLP中得到答案並與真實label做對比，如果得到的accuracy大於某個threshold則停止輪迴


#與真實label做對比，根據規則放入新data set中，或者放回重要性排序步驟中


#每一輪結束修改prompt，並使用dataset訓練NLP模型

