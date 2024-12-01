import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import spacy
from nltk.corpus import stopwords
import re
import openai
import numpy as np
import tensorflow_hub as hub

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()


# Initialize necessary components
openai.api_key = "your_openai_api_key"
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load BERT model for masked language modeling
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Utility functions
def spacy_tokenize(sentence):
    doc = nlp(sentence)
    return ' '.join([token.text for token in doc])


def mask_word_in_sentence(sentence, pos):
    words = sentence.split()
    if pos < len(words):
        words[pos] = '[MASK]'
    return ' '.join(words)


def get_word_importance(masked_sentences, original_sentences, pos_masks):
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


def is_valid_word(word, pos_tag):
    # Ensure word is valid (not a stop word, proper noun, or invalid POS)
    return (word.lower() not in stop_words and
            pos_tag not in ['PRON', 'DET', 'PUNCT', 'SYM'] and word.isalpha())


# Protect named entities and pronouns
def protect_named_entities(doc):
    protected_indices = set()
    for ent in doc.ents:
        for token in ent:
            protected_indices.add(token.i)
    return protected_indices


# Calculate semantic similarity using Universal Sentence Encoder
def calculate_use_similarity(original_sentence, modified_sentence):
    original_embedding = use_model([original_sentence])
    modified_embedding = use_model([modified_sentence])
    similarity = np.inner(original_embedding, modified_embedding)
    return similarity[0][0]


def get_synonym_from_gpt2(current_text, word):
    prompt = (f"Here are some examples of how to list synonyms:\n"
              f"Word: teacher\n1. instructor\n2. educator\n3. mentor\n4. tutor\n5. professor\n"
              f"Now, list 5 synonyms for the word '{word}' in the sentence: '{current_text}'."
              f"Only output the synonyms in the format:\n1. synonym 1\n2. synonym 2\n3. synonym 3\n4. synonym 4\n5. synonym 5")

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


import re


# Word replacement with constraints
def replace_top_30_percent_words(original_sentence, importance_scores, threshold=0.9):
    importance_scores_sorted = sorted(importance_scores, key=lambda x: x[1], reverse=True)
    num_words_to_replace = max(1, int(0.3 * len(importance_scores_sorted)))
    words_to_replace = [word for word, score in importance_scores[:num_words_to_replace]]
    replaced_sentence = []

    words_with_punctuation = re.findall(r'\w+|[^\w\s]', original_sentence)

    for token in words_with_punctuation:
        word_without_punct = re.sub(r'[^\w]', '', token)
        if word_without_punct in words_to_replace:
            synonym = get_synonym_from_gpt2(original_sentence, word_without_punct)
            print(word_without_punct)
            temp_sentence = original_sentence.replace(word_without_punct, synonym)

            # Use Universal Sentence Encoder to check semantic similarity
            if calculate_use_similarity(original_sentence, temp_sentence) >= threshold:
                replaced_token = token.replace(word_without_punct, synonym)
                print(replaced_token)
            else:
                replaced_sentence.append(token)
        else:
            replaced_sentence.append(token)

    return ' '.join(replaced_sentence)


# Load and process dataset
splits = {'train': 'plain_text/train-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["train"])
original_sentences = df.head(1)['text'].tolist()
processed_sentences = [spacy_tokenize(sentence) for sentence in original_sentences]

# Generate word importance rankings
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

    print(original_sentence_importance)
    original_sentence_importance.sort(key=lambda x: x[1])
    original_importance_scores.append(original_sentence_importance)

# Replace top 30% important words with synonyms
for i, original_sentence in enumerate(original_sentences):
    new_sentence = replace_top_30_percent_words(original_sentence, original_importance_scores[i])
    print(f"Original Sentence {i + 1}: {original_sentence}")
    print(f"New Sentence {i + 1} with LLM Synonym Replacement: {new_sentence}")
