import spacy

# first download the language model with python -m spacy download en_core_web_sm

en_tokenizer = spacy.load("en_core_web_sm")
de_tokenizer = spacy.load("de_core_news_sm")

tokens = en_tokenizer("Hello, world!")

print([token.text for token in tokens])
