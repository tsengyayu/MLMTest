import csv

import polars as pl
import ollama
from tqdm.notebook import tqdm
from transformers import BertTokenizer,AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import spacy
from spacy.util import compile_infix_regex
import torch
import re
import random
from sentence_transformers import SentenceTransformer, util

# 1. 先觀察語意相似度，再來是label，最後是confidence
# 當未通過標準時，隨機選擇一個情感給prompt，嘗試修改次數上限可為3，
# 2. 已經排除停用詞、主語、專有名詞、代詞，已使用LLM生成同義詞
# 3. 這部分需要再思考emoji要在什麼時候生成會比較好
# 4. 句法解析器是什麼要看一下
# 5. 暫時不考慮人類評估
# 6. 樣本個類別要平衡
# 8. 是否要用自動化提示調整
# 將剛才失敗的原因放進去

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

# 載入模型和 tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./kaggle/working/results/checkpoint-500')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embedding_model = AutoModel.from_pretrained("./kaggle/working/results/checkpoint-500")
simility_model = SentenceTransformer('all-MiniLM-L6-v2')

nlp = spacy.load("en_core_web_sm") #加載NLP模型

splits = {'train': 'data/train.jsonl', 'validation': 'data/validation.jsonl', 'test': 'data/test.jsonl'}
df_train = pl.read_ndjson('hf://datasets/AdamCodd/emotion-balanced/' + splits['train'])
df_train = df_train.to_pandas()
df_test = pl.read_ndjson('hf://datasets/AdamCodd/emotion-balanced/' + splits['test'])
df_test = df_test.to_pandas()

test_texts = df_test.head(5)['text'].tolist()
test_labels = df_test.head(5)['label'].tolist()

emotion_label = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}

#生成新句子
def add_new_sentence(original_sentence, new_label):
    # prompt = f"Based on the following sentence:'{original_sentence}', add a new sentence that naturally flows from the " \
    #          f"original while maintaining the same emotion. When the label is 0, the original sentence’s " \
    #          f"emotion is negative, and when the label is 1, the emotion is positive. " \
    #          f"The current label is '{original_label}'. Ensure that the added sentence deepens or clarifies the " \
    #          f"expressed emotion and complements the context effectively.The response only includes the final total sentence";

    # prompt = f"Please add a new sentence that has a faint hint of '{new_label}' after the following sentence：'{original_sentence}' to " \
    #          f"make it slightly more '{new_label}'.but keep the original meaning of the sentence and use only subtle adjustments in tone." \
    #          f"The response only includes the final total sentence.";
    prompt = f"Please add a new sentence that has an emotion that matches the original sentence after the following sentence：'{original_sentence}'." \
             f"It keep the original meaning of the sentence and use only subtle adjustments in tone." \
             f"The response only includes the final total sentence.";


    data = ollama.generate(model='data_processor', prompt=prompt)

    generated_text = data['response']

    return generated_text

def spacy_tokenize(sentence): # 句子標記化處理，分割成一系列詞語和標點符號
    doc = nlp(sentence)
    return ' '.join([token.text for token in doc])

#得到詞語重要性排序1
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

#得到詞語重要性排序2
def get_word_importance_new(sentence, tokenizer, model, embedding_model):
    # 將句子進行分詞
    words = sentence.split()

    # 對完整句子進行初始預測
    original_inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    original_outputs = model(**original_inputs)
    original_prob = torch.softmax(original_outputs.logits, dim=-1).max().item()

    # 計算整體句子的嵌入表示
    with torch.no_grad():
        original_embeddings = embedding_model(**original_inputs).last_hidden_state.mean(dim=1)  # 平均池化嵌入

    importance_scores = []

    for idx, word in enumerate(words):
        # 遮蔽當前詞語並重新生成嵌入
        masked_sentence = ' '.join([w if i != idx else '[MASK]' for i, w in enumerate(words)])
        inputs = tokenizer(masked_sentence, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            masked_embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)  # 平均池化嵌入

        # 計算遮蔽後的嵌入與原始嵌入的距離
        importance = torch.nn.functional.cosine_similarity(original_embeddings, masked_embeddings).item()
        importance_scores.append((word, 1 - importance))  # 1 - cosine similarity作為距離

    # 按重要性降序排列詞語
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    return importance_scores

#計算詞語之間的距離
def calculate_distance(word1, word2):
    # 使用 embedding_model 計算詞語嵌入
    inputs1 = tokenizer(word1, return_tensors="pt")
    inputs2 = tokenizer(word2, return_tensors="pt")

    with torch.no_grad():
        embedding1 = embedding_model(**inputs1).last_hidden_state.mean(dim=1)
        embedding2 = embedding_model(**inputs2).last_hidden_state.mean(dim=1)

    # 計算歐式距離
    distance = torch.dist(embedding1, embedding2).item()
    return distance

#計算語意相似度
def calculate_simility(original_sentence, modified_sentence, threshold=0.8):
    embedding1 = simility_model.encode(original_sentence, convert_to_tensor=True)
    embedding2 = simility_model.encode(modified_sentence, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    print("similarity: ", similarity)
    return similarity

#計算同義詞語意距離
def get_distant_synonym(current_text, word,new_label, try_time, problem, distance_threshold=0.8):
    synonyms = get_synonym_from_llama(current_text, word, new_label, try_time, problem)

    # 找到語義距離最遠的同義詞
    distant_synonym = word
    max_distance = 0

    for synonym in synonyms:
        distance = calculate_distance(word, synonym)

        if distance > max_distance and distance >= distance_threshold:
            max_distance = distance
            distant_synonym = synonym

    return distant_synonym

#生成同義詞
def get_synonym_from_llama(current_text, word, new_label, try_time, problem):
    # prompt = f"Given the following sentence: '{current_text} ', generate a list of replacement words for the word'{word}' in the sentence to confuse the sentiment classification model. The requirements are as follows:" \
    #          f"1. The replacement words should maintain grammatical consistency, especially for pronouns in terms of tense, person, etc." \
    #          f"2. The generated replacements should fit the sentence context and ensure that the sentence remains semantically similar, with natural word order and readability." \
    #          f"3. Focus on replacing descriptive words (e.g., adjectives, adverbs) and avoid changing core nouns or verbs." \
    #          f"4. Where possible, the replacements should confuse the model’s prediction, but the primary meaning and structure of the sentence should remain intact." \
    #          f"Please generate replacements for each word in the format:'1. replacement 1\n 2. replacement 2\n ... 10. replacement 10\n'.";

    # prompt = f"Please generate a list of replacement words that has a faint hint of '{new_label} ' for the word'{word}'to the following sentence：['{current_text} '] to " \
    #          f"make it slightly more '{new_label} ', but keep the original meaning of the sentence and use only subtle adjustments in tone." \
    #          f"Please generate replacements for each word in the format:'1. replacement 1\n 2. replacement 2\n ... 10. replacement 10\n'.";
    if(try_time==3):
        prompt = f"Please generate a list of 20 subtle replacement words for '{word}' in the following sentence: '{current_text}'. "\
            f"The replacements should subtly adjust the tone to better align with '{new_label}' while maintaining the sentence's original meaning. "\
            f"Ensure the replacements are natural, contextually appropriate, and not overly strong or extreme. '{problem}' "\
            f"Only provide the list of replacements in this format:\n"\
            f"1. subtle_replacement_1\n2. subtle_replacement_2\n...10. subtle_replacement_10"
        print("new label: ", new_label)
    # prompt = f"Please generate a list of subtle replacement words for '{word}' that adjust the tone of the sentence slightly. "\
    #     f"The replacements should not significantly change the meaning but should introduce a faint hint of a different emotion. "\
    #     f"Make sure the replacements are not overly strong or extreme. Provide 10 alternatives in this format:\n"\
    #     f"1. subtle_replacement_1\n2. subtle_replacement_2\n...10. subtle_replacement_10\n\n"\
    #     f"Context: '{current_text}'."
    if (try_time == 1 or try_time==2):
        # prompt = f"Please generate a list of 20 replacement words for '{word}' in the following sentence: '{current_text}'. " \
        #          f"The replacements should subtly adjust the tone to better align with a different emotion while maintaining the original meaning of the sentence. " \
        #          f"Focus on generating words that are emotionally resonant and contextually appropriate, avoiding overly neutral or unrelated terms. " \
        #          f"{problem}" \
        #          f"Provide the list of replacements in this format:  " \
        #          f"1. replacement_word_1  " \
        #          f"2. replacement_word_2  " \
        #          f"...  " \
        #          f"20. replacement_word_20  "
        prompt = (
            f"Please generate a list of 20 subtle replacement words for '{word}' in the following sentence: '{current_text}'. "
            f"The replacements should slightly adjust the tone, introducing a faint hint of a different emotion while maintaining the sentence's original meaning. "
            f"Ensure the replacements are natural, contextually appropriate, and not overly strong or extreme. '{problem}'"
            f"Only provide the list of replacements in this format:\n"
            f"1. subtle_replacement_1\n2. subtle_replacement_2\n...20. subtle_replacement_20"
        )


    data = ollama.generate(model='data_processor', prompt=prompt)
    # print("prompy: ",prompt)
    generated_text = data['response']
    # print("generated_synonym_from_llama: ", generated_text)
    synonyms = re.findall(r'\d+\.\s(.+)', generated_text)

    return synonyms if synonyms else word

#生成同意短語
def get_synonym_phrase_from_llama(current_text, word, new_label):
    prompt = f"Please generate a list of replacement phrase that has a faint hint of '{new_label} ' " \
             f"for the word'{word}'to the following sentence：['{current_text} '] to " \
             f"make it slightly more '{new_label} ', but keep the original meaning of the sentence and use only subtle adjustments in tone." \
             f"Please generate replacements for each word in the format:" \
             f"'1. replacement phrase 1\n 2. replacement phrase 2\n ... 5. replacement phrase 5\n'.";

    data = ollama.generate(model='data_processor', prompt=prompt)

    generated_text = data['response']
    # print("generated_synonym_phrase_from_llama: ", generated_text)
    synonyms = re.findall(r'\d\.\s(.+)', generated_text)


    return synonyms[0] if generated_text else word

#生成emoji
def get_emoji_from_llama(sentence, label):
    # prompt = f"Add a kaomoji before or after phrase '{word}' to express emotion." \
    #          f"For example,if original phrase is 'sense a deep-seated anger'. adding emoji, it can become '😄sense a deep-seated anger(°_°)'. " \
    #          f"But if original phrase is 'girl'. adding emoji, it can become '😄girl(°_°)'" \
    #          f"Please generate kaomoji for phrase and show total in the format:" \
    #          f"'1. replacement phrase 1\n 2. replacement phrase 2\n ... 5. replacement phrase 5.\n'."
    prompt = f"Please select an emoji that reflects a different emotion from the original label and add it to the end of the sentence without disrupting the overall context. " \
             f"Ensure the sentence's meaning remains consistent. Only return the modified sentence (including the chosen emoji).  " \
             f"Below are the sentence and its original label:  " \
             f"Original Label:  '{label}'" \
             f"Sentence:  '{sentence}'"

    data = ollama.generate(model='data_processor', prompt=prompt)

    generated_text = data['response']
    return generated_text
    # print("generated_emoji_from_llama: ", generated_text)
    # synonyms = re.findall(r'\d\.\s([^\n]+)', generated_text)

    # return synonyms[0] if generated_text else word

#檢查句子語法是否有誤
def check_sentence_from_llama(sentence, original_label):
    prompt = f"Check the following sentence for grammatical errors. If there are errors, correct them with the minimal necessary changes " \
             f"to keep the meaning and tone unchanged, while If there is an emoji in the sentence that can move the model's emotion away " \
             f"from the original emotion:'{original_label}', it is retained. Output format just includes changed sentence, only the corrected sentence. " \
             f"If no changes are needed, output the original sentence without any additional text:\n\n{sentence}"
    # prompt = f"Please check the following sentence for grammatical errors and correct them with the minimal necessary changes, ensuring the original meaning and tone are preserved. " \
    #          f"If the sentence contains emojis： " \
    #          f"1. Retain the emoji only if it meaningfully influences the sentence's emotion in a way that moves the emotional tone further away from the original emotion: '{original_label}'." \
    #          f"2. Remove the emoji if it causes the sentence to deviate too far from the original meaning or structure.  " \
    #          f"Output only the corrected sentence. If no changes are needed, output the original sentence without any additional text.  " \
    #          f"Sentence to check: '{sentence}'"

    data = ollama.generate(model='data_processor', prompt=prompt)

    generated_text = data['response']
    return generated_text

#比較同義詞和同義短語的信心程度
def compare_word_and_phrase(sentence, word, synonym, synonym_phrase):
    # 計算同義詞替換的信心分數
    sentence_with_synonym = sentence.replace(word, synonym)
    encoding_synonym = tokenizer(sentence_with_synonym, return_tensors="pt")
    logits_synonym = model(**encoding_synonym).logits
    confidence_synonym = torch.softmax(logits_synonym, dim=-1).max().item()
    # print("1: ",confidence_synonym)

    # 計算同義短語替換的信心分數
    sentence_with_phrase = sentence.replace(word, synonym_phrase)
    encoding_phrase = tokenizer(sentence_with_phrase, return_tensors="pt")
    logits_phrase = model(**encoding_phrase).logits
    confidence_phrase = torch.softmax(logits_phrase, dim=-1).max().item()
    # print("2: ",confidence_phrase)

    # 選擇信心分數較低的替換
    if confidence_synonym < confidence_phrase:
        return synonym
    else:
        return synonym_phrase

#替換前10%的詞語
def replace_top_10_percent_words(original_sentence, importance_scores, new_label, try_time, problem):
    importance_scores_sorted = sorted(importance_scores, key=lambda x: x[1], reverse=True)
    num_words_to_replace = max(1, int(0.1 * len(importance_scores_sorted)))
    words_to_replace = [word for word, score in importance_scores[:num_words_to_replace]]
    replaced_sentence = []

    # 解析原始句子以識別人名、地名等
    doc = nlp(original_sentence)
    proper_nouns = {ent.text for ent in doc.ents}  # 獲取人名、地名等命名實體
    pronouns = {token.text for token in doc if token.pos_ == "PRON"}  # 獲取所有代詞

    # 保留句子中的標點符號，分割成單詞與標點
    words_with_punctuation = re.findall(r'\w+|[^\w\s]', original_sentence)

    for token in words_with_punctuation:
        word_without_punct = re.sub(r'[^\w]', '', token)

        # 檢查是否為代詞或命名實體，不替換
        if word_without_punct in words_to_replace and word_without_punct not in proper_nouns and word_without_punct not in pronouns:
            synonym = get_distant_synonym(original_sentence, word_without_punct,new_label, try_time, problem)
            # synonym_phrase = get_synonym_phrase_from_llama(original_sentence, word_without_punct,new_label)
            # match = re.search(r'replacement phrase:\s*(.*)', synonym_phrase, re.IGNORECASE)
            # synonym_phrase_match = match.group(1).strip() if match else synonym
            # synonym_replaced = compare_word_and_phrase(original_sentence, word_without_punct, synonym, synonym_phrase)
            synonym_replaced = synonym
            # print("synonym without emoji: ", synonym_replaced)
            # if(try_time==2 or try_time==3):
            #     synonym_replaced = get_emoji_from_llama(synonym_replaced)
            # print("word_without_punct: ", word_without_punct)
            # print("synonym: ", synonym)
            # print("synonym_phrase: ", synonym_phrase)
            # print("synonym: ", synonym_replaced)
            replaced_token = token.replace(word_without_punct, synonym_replaced)
            replaced_sentence.append(replaced_token)
            # break;
        else:
            replaced_sentence.append(token)

    return ' '.join(replaced_sentence)

adversarial_samples = []
output_csv = "adversarial_samples.csv"


for i, (original_sentence, original_label) in enumerate(zip(test_texts, test_labels)):
    print(f"\nProcessing Sentence {i + 1}...")
    print("original sentence: ", original_sentence)
    # 創建排除原始標籤的標籤清單
    new_labels = [label for label in emotion_label if label != original_label]
    # 隨機選擇一個新的標籤
    new_label = emotion_label[random.choice(new_labels)]
    #生成新句子
    original_sentence = add_new_sentence(original_sentence, new_label)
    print("add_original sentence: ", original_sentence)

    #斷句
    # 停用會導致縮寫詞分開的 infix 模式
    infix_re = compile_infix_regex(nlp.Defaults.infixes)
    infix_patterns = [pattern for pattern in infix_re.pattern.split("|") if "'" not in pattern]
    nlp.tokenizer.infix_finditer = compile_infix_regex(infix_patterns).finditer
    processed_sentence = spacy_tokenize(original_sentence)

    problem = ""

    for try_time in range(1, 4):
        # 計算詞語重要性
        #方法一
        original_sentence_importance = get_word_importance(processed_sentence, tokenizer, model)
        # print("original_importance_scores", original_sentence_importance)
        #方法二：嵌入法
        # original_sentence_importance = get_word_importance_new(processed_sentence, tokenizer, model, embedding_model)
        # print("original_importance_scores", original_sentence_importance)

        # 生成同義詞替換句子
        new_sentence = replace_top_10_percent_words(original_sentence, original_sentence_importance, new_label, try_time, problem)
        if (try_time == 2 or try_time == 3):
            new_sentence = get_emoji_from_llama(new_sentence, emotion_label[original_label])
        new_sentence = check_sentence_from_llama(new_sentence, emotion_label[original_label])
        print("checked sentence: ", new_sentence)
        similarity = calculate_simility(original_sentence, new_sentence)

    #測試
        # new_sentence = "I feel overjoyed to see so many people participating! It’s wonderful to witness this celebration of art and culture, and the energy is so uplifting."
        # # 將測試句子進行 tokenization
        test_encodings = tokenizer(new_sentence, truncation=True, padding=True, return_tensors="pt")

        # 將測試數據傳入模型，進行預測
        outputs = model(**test_encodings)

        # 取得預測的標籤 (logits 是模型的輸出)
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)
        # 計算信心分數
        confidence = torch.softmax(logits, dim=-1).max().item()
        print(f"Predicted Confidence Scores: {confidence}")

        # 轉換為可讀的標籤
        print("original labels:", original_label)
        print("Predicted labels:", predictions.item())
        if(similarity > 0.8 and (predictions.item() != original_label or confidence <= 0.6)):
            print("Success! try time: ", try_time)
            adversarial_samples.append({
                "original_sentence": original_sentence,
                "original_label": original_label,
                "new_sentence": new_sentence,
                "new_label": predictions.item()
            })
            break
        #添加問題到prompt
        if(similarity < 0.8):
            problem = "The previously generated sentence failed semantic similarity checks (similarity: "+str(similarity)+"). " \
                      "Example of a failed sentence: 「"+new_sentence+"」. Avoid using words from the failed sentence and regenerate " \
                      "replacements based on the original sentence. "
        elif(predictions.item() == original_label or confidence > 0.7):
            problem = "The sentence labels generated in the previous round did not change. Example of a failed sentence: 「"+new_sentence+"」. " \
                      "This time, the generated words should be more effective in shifting the emotional tone or direction while keeping the " \
                      "sentence natural and contextually appropriate. "

with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["original_sentence", "original_label", "new_sentence", "new_label"])
    writer.writeheader()
    writer.writerows(adversarial_samples)

print(f"Adversarial samples saved to {output_csv}")

    # # prompt = f'Please add a faint hint of worry to the following sentence：'+new_sentence+' to make it slightly more fearful, and replace all the emotion words in the sentence with other words, but keep the original meaning of the sentence and use only subtle adjustments in tone.';
    # prompt = f'Please add a faint hint of '+emotion_label[new_label]+' to the following sentence：['+new_sentence+'] to make it slightly more '+emotion_label[new_label]+',and replace all the emotion words in the sentence with other words, but keep the original meaning of the sentence and use only subtle adjustments in tone.'
    # data = ollama.generate(model='data_processor', prompt=prompt)
    # print("prompt: ", prompt)
    # generated_text = data['response']
    # print("generated_text_from_llama: ", generated_text)
    #
    # # 將測試句子進行 tokenization
    # test_encodings = tokenizer(generated_text, truncation=True, padding=True, return_tensors="pt")
    #
    # # 將測試數據傳入模型，進行預測
    # outputs = model(**test_encodings)
    #
    # # 取得預測的標籤 (logits 是模型的輸出)
    # logits = outputs.logits
    # predictions = logits.argmax(dim=-1)
    #
    # # 計算信心分數
    # confidence = torch.softmax(logits, dim=-1).max().item()
    # print(f"Predicted Confidence Scores: {confidence}")
    #
    # print("Predicted labels:", predictions.item())

# 是否需要把問題放進prompt中!!
# 要把檢查後可能會出現的多餘內容去掉
# 1.正常 2.增加emoji 3.增加隨機label ？語意相似度太低的解決方法？
# 一次性進行過多的修改可能導致語意損失