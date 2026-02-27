from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./local_model")
tokenizer.save_pretrained("./local_model")