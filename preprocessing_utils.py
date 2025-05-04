import spacy
import re

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# MATH_SYMBOLS = set("+-=*/^%<>!~|()[]}{")

# def clean_and_lemmatize_keep_mathsym(docs):
#     # Lemmatizes, removes numbers, stop words and spaces. However, keeps these "+-=*/^%<>!~|()[]\}{" Math Symbols.  
#     cleaned_docs = []
#     for doc in docs:
#         doc = doc.lower()
#         doc = re.sub(r'\d+', '', doc)
#         spacy_doc = nlp(doc)
#         tokens = []
#         for token in spacy_doc:
#             if token.text in MATH_SYMBOLS:
#                 tokens.append(token.text)
#             elif token.is_alpha and not token.is_stop and not token.is_space:
#                 tokens.append(token.lemma_)
#         cleaned_docs.append(" ".join(tokens))
#     return cleaned_docs


def clean_and_lemmatize(docs):
    # Lemmatizes, Removes numbers, stop words, punctuations and spaces
    cleaned_docs = []
    for doc in docs:
        doc = doc.lower()
        doc = re.sub(r'\d+', '', doc)
        spacy_doc = nlp(doc)
        tokens = [token.lemma_ for token in spacy_doc
                      if not token.is_stop and not token.is_punct and not token.is_space]
        cleaned_docs.append(" ".join(tokens))
    return cleaned_docs