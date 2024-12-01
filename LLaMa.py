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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM

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

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #加載BERT模型用於MLM
# mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased")

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

# def get_word_importance(masked_sentences, original_sentences, pos_masks): #對被遮罩的句子，使用MLM計算每個詞的重要性
#     encoded_inputs = tokenizer(masked_sentences, return_tensors='pt', padding='max_length', max_length=128)
#     outputs = model(**encoded_inputs)
#
#     original_probs = []
#     for i, pos in enumerate(pos_masks):
#         if pos < len(original_sentences[i].split()):
#             original_token = tokenizer.tokenize(original_sentences[i].split()[pos])
#             original_token_id = tokenizer.convert_tokens_to_ids(original_token[0])
#             original_prob = torch.softmax(outputs.logits[i, pos, :], dim=-1)[original_token_id].item()
#             original_probs.append(original_prob)
#
#     return original_probs
def get_word_importance(sentence, tokenizer, model):
    # 將句子分詞
    words = sentence.split()
    # 對完整句子進行初始預測
    original_inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    original_outputs = model(**original_inputs)
    original_prob = torch.softmax(original_outputs.logits, dim=-1).max().item()

    importance_scores = []

    for idx, word in enumerate(words):
        # 將詞語替換為 [MASK] 進行再預測
        masked_sentence = ' '.join([w if i != idx else '[MASK]' for i, w in enumerate(words)])
        inputs = tokenizer(masked_sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        new_prob = torch.softmax(outputs.logits, dim=-1).max().item()

        # 計算重要性作為概率變化
        importance = abs(original_prob - new_prob)
        importance_scores.append((word, importance))

    # 按重要性降序排列詞語
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    return importance_scores



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
    # prompt = f"Given the following sentence:'{current_text} ', find a list of k synonyms for the word'{word}' within the sentence. This sentence's label is negative." \
    #          f"When replacing the word with its synonyms, the new sentence should maintain the semantic meaning, syntax, and grammar " \
    #          f"of the original sentence. But this new sentence needs to confuse the model as much as possible, so that the model output result is opposite to the original emotion classification result. The new result should be changed to positive." \
    #          f"You should only output the synonyms of the word but not the sentences containing them. " \
    #          f"You should also format your output as following: '1. word 1\n 2. word 2\n 3. word 3\n ... 5. word 5\n'. " \
    #          f"Furthermore, you should rank your results by the semantic similarity between the original sentence and " \
    #          f"the sentence after replacing the word with its synonyms.";
    prompt = f"Given the following sentence: '{current_text} ', generate a list of replacement words for the word'{word}' in the sentence to confuse the sentiment classification model. The requirements are as follows:" \
             f"1. The replacement words should maintain grammatical consistency, especially for pronouns in terms of tense, person, etc." \
             f"2. The generated replacements should fit the sentence context and ensure that the sentence remains semantically similar, with natural word order and readability." \
             f"3. Focus on replacing descriptive words (e.g., adjectives, adverbs) and avoid changing core nouns or verbs." \
             f"4. Where possible, the replacements should confuse the model’s prediction, but the primary meaning and structure of the sentence should remain intact." \
             f"Please generate replacements for each word in the format:'1. replacement 1\n 2. replacement 2\n ... 5. replacement 5\n'.";

    data = ollama.generate(model='data_processor', prompt=prompt)

    generated_text = data['response']
    # print("generated_text: ", generated_text)
    synonyms = re.findall(r'\d\.\s(\w+)', generated_text)

    return synonyms[0] if synonyms else word

def get_sentence_from_llama(current_text):
    prompt = f"Please help me check if there are any grammatical errors in this sentence:'{current_text}'. After correcting the errors, " \
             f"only the revised sentence is required to return the content.";

    data = ollama.generate(model='data_processor', prompt=prompt)

    generated_text = data['response']

    return generated_text

##note:主語不應該替換、
# def replace_top_30_percent_words(original_sentence, importance_scores, threshold=0.9):
#     importance_scores_sorted = sorted(importance_scores, key=lambda x: x[1], reverse=True)
#     num_words_to_replace = max(1, int(0.3 * len(importance_scores_sorted)))
#     words_to_replace = [word for word, score in importance_scores[:num_words_to_replace]]
#     replaced_sentence = []
#
#     words_with_punctuation = re.findall(r'\w+|[^\w\s]', original_sentence)
#
#     for token in words_with_punctuation:
#         word_without_punct = re.sub(r'[^\w]', '', token)
#         if word_without_punct in words_to_replace:
#             synonym = get_synonym_from_llama(original_sentence, word_without_punct)
#             # print("synonym: ",synonym)
#             temp_sentence = original_sentence.replace(word_without_punct, synonym)
#
#             # Use Universal Sentence Encoder to check semantic similarity
#             if calculate_use_similarity(original_sentence, temp_sentence) >= threshold:
#                 replaced_token = token.replace(word_without_punct, synonym)
#                 replaced_sentence.append(replaced_token)
#             else:
#                 replaced_sentence.append(token)
#         else:
#             replaced_sentence.append(token)
#
#     return ' '.join(replaced_sentence)

def replace_top_30_percent_words(original_sentence, importance_scores, threshold=0.9):
    # 排序詞語重要性分數，取前30%的詞語
    importance_scores_sorted = sorted(importance_scores, key=lambda x: x[1], reverse=True)
    num_words_to_replace = max(1, int(0.3 * len(importance_scores_sorted)))
    words_to_replace = [word for word, score in importance_scores[:num_words_to_replace]]
    replaced_sentence = []

    # 保留句子中的標點符號，分割成單詞與標點
    words_with_punctuation = re.findall(r'\w+|[^\w\s]', original_sentence)

    for token in words_with_punctuation:
        word_without_punct = re.sub(r'[^\w]', '', token)

        # 若詞語在替換列表中，則替換為空白
        if word_without_punct in words_to_replace:
            temp_sentence = original_sentence.replace(word_without_punct, "")  # 替換為空白

            # 檢查替換後的句子語意相似性
            if calculate_use_similarity(original_sentence, temp_sentence) >= threshold:
                replaced_sentence.append("")  # 替換為空白
            else:
                replaced_sentence.append(token)
        else:
            replaced_sentence.append(token)

    # 將替換後的句子合併並返回
    return ' '.join(replaced_sentence)


def get_label(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()

    # 打印置信度分數（對應每個類別的概率）
    confidence_scores = predictions.squeeze().tolist()  # 轉換為 list
    print(f"Confidence Scores: {confidence_scores}")
    return predicted_class, confidence_scores

#載入資料
splits = {'train': 'plain_text/train-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["train"])

#MLM詞語重要性排序
original_sentences = df.head(5)['text'].tolist() #取第一筆數據
original_labels = df.head(5)['label'].tolist()

# processed_sentences = [spacy_tokenize(sentence) for sentence in original_sentences]
# print("processed_sentences", processed_sentences)
# pos_masks = [{'original': list(range(len(sentence.split())))} for sentence in processed_sentences]
# original_importance_scores = []
#
# for i, sentence in enumerate(processed_sentences):
#     doc_original = nlp(sentence)
#     protected_indices = protect_named_entities(doc_original)
#     original_sentence_importance = []
#
#     for pos in pos_masks[i]['original']:
#         if pos not in protected_indices and pos < len(sentence.split()):
#             word = sentence.split()[pos]
#             pos_tag = doc_original[pos].pos_
#             if is_valid_word(word, pos_tag):
#                 masked_sentence = mask_word_in_sentence(sentence, pos)
#                 original_prob = get_word_importance([masked_sentence], [sentence], [pos])[0]
#                 original_sentence_importance.append((word, original_prob))
#
#     original_sentence_importance.sort(key=lambda x: x[1])
#     original_importance_scores.append(original_sentence_importance)
#     print("original_importance_scores", original_importance_scores)
#
# #LLM生成同義詞
# for i, (original_sentence, original_label) in enumerate(zip(original_sentences, original_labels)):
#     new_sentence = replace_top_30_percent_words(original_sentence, original_importance_scores[i])
#     print(f"Original Sentence {i + 1}: {original_sentence}")
#     print(f"New Sentence {i + 1} with LLM Synonym Replacement: {new_sentence}")
#
#     #獲取替換句子的標籤
#     # original_label = get_label(original_sentence)
#     new_label = get_label(new_sentence)
#
#     print(f"Original Label: {original_label}, New Label: {new_label}")
#
#     if original_label == new_label:
#         print("Label matches: True")
#     else:
#         print("Label matches: False")

# 預處理每個句子，進行詞語重要性排序和同義詞替換
# for i, (original_sentence, original_label) in enumerate(zip(original_sentences, original_labels)):
#     print(f"\nProcessing Sentence {i + 1}...")
#     processed_sentence = spacy_tokenize(original_sentence)
#
#     # 計算詞語重要性
#     pos_mask = list(range(len(processed_sentence.split())))
#     doc_original = nlp(processed_sentence)
#     protected_indices = protect_named_entities(doc_original)
#     original_sentence_importance = []
#
#     for pos in pos_mask:
#         if pos not in protected_indices and pos < len(processed_sentence.split()):
#             word = processed_sentence.split()[pos]
#             pos_tag = doc_original[pos].pos_
#             if is_valid_word(word, pos_tag):
#                 masked_sentence = mask_word_in_sentence(processed_sentence, pos)
#                 original_prob = get_word_importance([masked_sentence], [processed_sentence], [pos])[0]
#                 original_sentence_importance.append((word, original_prob))
#
#     original_sentence_importance.sort(key=lambda x: x[1], reverse=True)
#     print("original_importance_scores", original_sentence_importance)
#
#     # 生成同義詞替換句子
#     new_sentence = replace_top_30_percent_words(original_sentence, original_sentence_importance)
#     print(f"Original Sentence: {original_sentence}")
#     print(f"New Sentence with LLM Synonym Replacement: {new_sentence}")
#
#     # 獲取替換句子的標籤
#     new_label = get_label(new_sentence)
#
#     print(f"Original Label: {original_label}, New Label: {new_label}")
#     if original_label == new_label:
#         print("Label matches: True")
#     else:
#         print("Label matches: False")

for i, (original_sentence, original_label) in enumerate(zip(original_sentences, original_labels)):
    print(f"\nProcessing Sentence {i + 1}...")
    processed_sentence = spacy_tokenize(original_sentence)

    # 計算詞語重要性
    original_sentence_importance = get_word_importance(processed_sentence, tokenizer, model)
    print("original_importance_scores", original_sentence_importance)

    # 生成同義詞替換句子
    new_sentence = replace_top_30_percent_words(original_sentence, original_sentence_importance)
    print(f"Original Sentence: {original_sentence}")
    print(f"New Sentence with LLM Synonym Replacement: {new_sentence}")

    # new_sentence = get_sentence_from_llama(new_sentence)
    # print(f"New Sentence with LLM Synonym Replacement: {new_sentence}")

    # 獲取替換句子的標籤和置信度
    original_label, original_confidence_scores = get_label(original_sentence)
    new_label, confidence_scores = get_label(new_sentence)

    print(f"Original Label: {original_label}, New Label: {new_label}")
    print(f"Confidence Scores for old Label: {original_confidence_scores}")
    print(f"Confidence Scores for New Label: {confidence_scores}")

    if original_label == new_label:
        print("Label matches: True")
    else:
        print("Label matches: False")

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

