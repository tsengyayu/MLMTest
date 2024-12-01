# import pandas as pd
# import torch
# from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
#
# # 讀取資料集
# splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet'}
# df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["train"])
#
# # 提取原始句子和標籤
# original_sentences = df.head(1)['text'].tolist()  # 從資料集中提取 'text' 欄位作為句子
#
# # 載入預訓練的 BERT 模型和 tokenizer
# enc = BertTokenizer.from_pretrained('bert-base-uncased')
# mlm_model_ts = BertForMaskedLM.from_pretrained('bert-base-uncased')
#
# # 用於計算詞語重要性
# def get_word_importance(masked_sentences, original_sentences, pos_masks):
#     # 編碼句子
#     encoded_inputs = enc(masked_sentences, return_tensors='pt', padding='max_length', max_length=128)
#     outputs = mlm_model_ts(**encoded_inputs)
#
#     # 計算原始詞的預測分數
#     original_probs = []
#     for i, pos in enumerate(pos_masks):
#         original_token = enc.tokenize(original_sentences[i].split()[pos])  # 原始詞 token
#         original_token_id = enc.convert_tokens_to_ids(original_token[0])  # 取首個子詞的 token id
#         original_prob = torch.softmax(outputs.logits[i, pos, :], dim=-1)[original_token_id].item()
#         original_probs.append(original_prob)
#
#     return original_probs
#
# # 用於生成掩碼句子
# def mask_word_in_sentence(sentence, pos):
#     words = sentence.split()
#     words[pos] = '[MASK]'
#     return ' '.join(words)
#
# # 設定生成的參數
# max_length = 1024  # 生成文本的最大長度
# num_return_sequences = 1  # 每個輸入句子生成幾個樣本
#
# # 載入 GPT-2 模型和 Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
#
# # 存放原始句子和生成樣本的列表
# all_sentences = []
#
# # 檢查 input_ids 長度並調整生成長度，避免超過 GPT-2 限制
# for sentence in original_sentences:
#     input_ids = tokenizer.encode(sentence, return_tensors='pt')
#
#     if input_ids.size(-1) == 0:
#         print(f"Error: Input sentence '{sentence}' is too short or improperly encoded.")
#         continue
#
#     if input_ids.size(-1) > 1024:
#         input_ids = input_ids[:, :1024]
#
#     outputs = model.generate(
#         input_ids,
#         max_length=min(1024, input_ids.size(-1) + 100),  # 確保不超過 1024
#         num_return_sequences=num_return_sequences,
#         do_sample=True,
#         top_p=0.95,
#         temperature=0.7,
#         pad_token_id=tokenizer.eos_token_id
#     )
#
#     generated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     all_sentences.append({'original': sentence, 'generated': generated_sentence})
#     print(generated_sentence)
#
# # 根據每個句子的長度生成詞語位置
# pos_masks = []
# for sentence_pair in all_sentences:
#     # 使用原始句子和生成句子，分別建立詞語位置列表
#     num_words_original = len(sentence_pair['original'].split())
#     num_words_generated = len(sentence_pair['generated'].split())
#
#     # 只掩碼詞語的有效索引
#     pos_masks.append({
#         'original': list(range(num_words_original)),
#         'generated': list(range(num_words_generated))
#     })
#
# # 查看生成的 pos_masks
# for i, pos_mask in enumerate(pos_masks):
#     print(f"Sentence {i + 1} pos_mask: {pos_mask}")
#
# # 計算原始和生成句子的詞語重要性
# original_importance_scores = []
# generated_importance_scores = []
#
# for i, sentence_pair in enumerate(all_sentences):
#     # 對原始句子計算詞語重要性
#     original_sentence_importance = []
#     for pos in pos_masks[i]['original']:
#         masked_sentence = mask_word_in_sentence(sentence_pair['original'], pos)
#         masked_sentences = [masked_sentence]
#         original_prob = get_word_importance(masked_sentences, [sentence_pair['original']], [pos])[0]
#         original_sentence_importance.append((sentence_pair['original'].split()[pos], original_prob))
#
#     original_sentence_importance.sort(key=lambda x: x[1])
#     original_importance_scores.append(original_sentence_importance)
#
#     # 對生成句子計算詞語重要性
#     generated_sentence_importance = []
#     for pos in pos_masks[i]['generated']:
#         masked_sentence = mask_word_in_sentence(sentence_pair['generated'], pos)
#         masked_sentences = [masked_sentence]
#         generated_prob = get_word_importance(masked_sentences, [sentence_pair['generated']], [pos])[0]
#         generated_sentence_importance.append((sentence_pair['generated'].split()[pos], generated_prob))
#
#     generated_sentence_importance.sort(key=lambda x: x[1])
#     generated_importance_scores.append(generated_sentence_importance)
#
# # 輸出每個句子中詞語的重要性排序
# for i, (original_importance, generated_importance) in enumerate(
#         zip(original_importance_scores, generated_importance_scores)):
#     print(f"Sentence {i + 1} original word importance ranking:")
#     for word, score in original_importance:
#         print(f"{word}: {score}")
#     print()
#
#     print(f"Sentence {i + 1} generated word importance ranking:")
#     for word, score in generated_importance:
#         print(f"{word}: {score}")
#     print()
# Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# tokenizer = AutoTokenizer.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")
# model = AutoModelForSequenceClassification.from_pretrained("randellcotta/distilbert-base-uncased-finetuned-yelp-polarity")
import pandas as pd
splits = {'train': 'plain_text/train-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["train"])
# original_sentences = df.head(1)['text'].tolist()
print(df)

