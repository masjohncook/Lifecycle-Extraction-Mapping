# from bs4 import BeautifulSoup
# from bs4.element import Comment
# from urllib.request import Request, urlopen #urllib has been split across
#
#
# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True
#
#
# def text_from_html(body):
#     soup = BeautifulSoup(body, 'html.parser')
#     texts = soup.findAll(text=True)
#     visible_texts = filter(tag_visible, texts)
#     return u" ".join(t.strip() for t in visible_texts)
# link = "http://www.nytimes.com/2009/12/21/us/21storm.html"
# req = Request(
#                 url=link,
#                 headers={'User-Agent': 'Chrome/116.0.0.0'}
#             )
#
# html =urlopen(req).read()
# text = text_from_html(html)
#
# #html = open('21storm.html').read()
# soup = BeautifulSoup(html)
# [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
# visible_text = soup.getText()
# print(text)

import trafilatura

# link = "https://outrunsec.com/2021/06/21/cyberseclabs-red-walkthrough/"
# link = "https://blog.csdn.net/weixin_41091001/article/details/124210151"

link = "https://www.hackingarticles.in/escalate_linux-vulnhub-walkthrough-part-1/"
downloaded = trafilatura.fetch_url(link)
text = trafilatura.extract(downloaded)
#print (text)


import nltk.data
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



combine = tokenizer.tokenize(text)
print ("Combined 6: ", combine[6])
print("\n\n\n\n\n\n")
doc = nlp(combine[6])
# Extract individual sentences
sentences = [sent.text for sent in doc.sents]

print("Separated sentences:")
for sentence in sentences:
    print(sentence)




# index = 0
# for i in combine:
#     print ("Sentence {d}".format(d=index), i)
#     index += 1

