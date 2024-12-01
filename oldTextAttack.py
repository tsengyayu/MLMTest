from oldTextAttack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加載預訓練的BERT模型和tokenizer
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

# 包裝成TextAttack的模型格式
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
