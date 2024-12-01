import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import spacy
from nltk.corpus import stopwords
import re
# import openai
import numpy as np
import tensorflow_hub as hub
# from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
import ollama
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

modelfile = """
FROM llama3
PARAMETER temperature 0.2
PARAMETER num_ctx 2048
SYSTEM 你是資料處裡的工具，你只會用英文回覆資料內容，不提供其他資訊
"""

##llama 3.2
ollama.create(model='data_processor', modelfile=modelfile)
ollama.list()
tqdm.pandas()

# openai.api_key = "your_openai_api_key"
nlp = spacy.load("en_core_web_sm") #加載NLP模型
stop_words = set(stopwords.words('english')) #停用詞

# 載入 LLaMa 模型和 tokenizer
# llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-2-7b-hf")
# llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/LLaMA-2-7b-hf")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #加載BERT模型用於MLM
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") #計算語意相似性

#已使用yelp dataset的bert模型
tokenizer = AutoTokenizer.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")
model = AutoModelForSequenceClassification.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")

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

def get_synonym_from_llama(current_text, word):
    # prompt = (f"Here are some examples of how to list synonyms:\n"
    #           f"Word: teacher\n1. instructor\n2. educator\n3. mentor\n4. tutor\n5. professor\n"
    #           f"Now, list 5 synonyms for the word '{word}' in the sentence: '{current_text}'."
    #           f"Only output the synonyms in the format:\n1. synonym 1\n2. synonym 2\n3. synonym 3\n4. synonym 4\n5. synonym 5")
    prompt = f"Given the following sentence:'{current_text} ', find a list of k synonyms for the word'{word}' within the sentence. This sentence's label is negative." \
             f"When replacing the word with its synonyms, the new sentence should maintain the semantic meaning, syntax, and grammar " \
             f"of the original sentence. But this new sentence needs to confuse the model as much as possible, so that the model output result is opposite to the original emotion classification result. The new result should be changed to positive." \
             f"You should only output the synonyms of the word but not the sentences containing them. " \
             f"You should also format your output as following: '1. word 1\n 2. word 2\n 3. word 3\n ... 5. word 5\n'. " \
             f"Furthermore, you should rank your results by the semantic similarity between the original sentence and " \
             f"the sentence after replacing the word with its synonyms.";

    data = ollama.generate(model='data_processor', prompt=prompt)

    generated_text = data['response']
    synonyms = re.findall(r'\d\.\s(\w+)', generated_text)

    return synonyms[0] if synonyms else word

def replace_with_synonyms_and_structure(original_sentence, importance_scores, threshold=0.9):
    # 提高替換比例
    importance_scores_sorted = sorted(importance_scores, key=lambda x: x[1], reverse=True)
    num_words_to_replace = max(1, int(0.5 * len(importance_scores_sorted)))
    words_to_replace = [word for word, score in importance_scores[:num_words_to_replace]]
    replaced_sentence = []

    words_with_punctuation = re.findall(r'\w+|[^\w\s]', original_sentence)

    for token in words_with_punctuation:
        word_without_punct = re.sub(r'[^\w]', '', token)
        if word_without_punct in words_to_replace:
            # 使用 LLM 生成同義詞
            synonym = get_synonym_from_llama(original_sentence, word_without_punct)
            print("synonym: ", synonym)
            temp_sentence = original_sentence.replace(word_without_punct, synonym)

            # 檢查語意相似性
            if calculate_use_similarity(original_sentence, temp_sentence) >= threshold:
                replaced_token = token.replace(word_without_punct, synonym)
                replaced_sentence.append(replaced_token)
            else:
                replaced_sentence.append(token)
        else:
            replaced_sentence.append(token)

    # 隨機更改語序
    if np.random.rand() > 0.5:
        replaced_sentence = replaced_sentence[::-1]  # 隨機顛倒句子順序

    return ' '.join(replaced_sentence)

def get_label(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()
    return predicted_class

#載入資料
splits = {'train': 'plain_text/train-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["train"])

#MLM詞語重要性排序
original_sentences = df.head(1)['text'].tolist() #取第一筆數據
original_labels = df.head(1)['label'].tolist()

# 更新主程式循環來使用新的替換函數
for i, (original_sentence, original_label) in enumerate(zip(original_sentences, original_labels)):
    print(f"\nProcessing Sentence {i + 1}...")
    processed_sentence = spacy_tokenize(original_sentence)

    # 計算詞語重要性
    pos_mask = list(range(len(processed_sentence.split())))
    doc_original = nlp(processed_sentence)
    protected_indices = protect_named_entities(doc_original)
    original_sentence_importance = []

    for pos in pos_mask:
        if pos not in protected_indices and pos < len(processed_sentence.split()):
            word = processed_sentence.split()[pos]
            pos_tag = doc_original[pos].pos_
            if is_valid_word(word, pos_tag):
                masked_sentence = mask_word_in_sentence(processed_sentence, pos)
                original_prob = get_word_importance([masked_sentence], [processed_sentence], [pos])[0]
                original_sentence_importance.append((word, original_prob))

    original_sentence_importance.sort(key=lambda x: x[1], reverse=True)
    print("original_importance_scores", original_sentence_importance)

    # 使用更高比例替換生成同義詞
    new_sentence = replace_with_synonyms_and_structure(original_sentence, original_sentence_importance)
    print(f"Original Sentence: {original_sentence}")
    print(f"New Sentence with LLM Synonym Replacement: {new_sentence}")

    # 獲取替換句子的標籤
    new_label = get_label(new_sentence)

    print(f"Original Label: {original_label}, New Label: {new_label}")
    if original_label == new_label:
        print("Label matches: True")
    else:
        print("Label matches: False")