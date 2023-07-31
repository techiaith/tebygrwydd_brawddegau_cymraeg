import spacy
from collections import Counter
import random

# Use this model: https://github.com/techiaith/spacy_cy_tag_lem_ner_lg
nlp = spacy.load("cy_techiaith_tag_lem_ner_lg")

sents_str = """Soniodd yr athro ei fod wedi gweld y disgybl yn taro'r ferch. 
Fe ddywedodd yr athrawes iddi sylwi ar y plentyn yn pwnio ei chwaer. 
Roedd y ci wedi bwyta'r olaf o'u bwyd pan oedden nhw'n cysgu. 
A hwythau yn eu gwely, bwytaodd y gath sbarion y pastai cig eidion. 
Dywedodd llefarydd ar ran yr undeb nad oedd modd iddynt dderbyn y cynnig. 
Gwrthododd y gweithwyr gynnig y cyflogwyr. 
Defnyddiodd lygoden i reoli'r cyfrifiadur. 
Hoffai ddefnyddio llygoden pan yn chwarae gemau cyfrifiadurol. 
Dyma'r rhagolygon ar gyfer y tywydd. 
Yfory bydd hi'n bwrw glaw dros y bryniau. 
Llithrodd ar ei chefn gan anafu ei hysgwydd. 
Disgynnodd yn boenus ar y llawr. 
Cyhoeddodd yr arlywydd gyfres newydd o sancsiynau yn erbyn Iran. 
Bydd yr Unol Daleithiau yn gwahardd masnachu ag Irac o hyn allan. 
Aeth am gyfweliad gyda'r cwmni ar ôl gweld yr hysbyseb yn papur newydd. 
Roedd swydd ddisgrifiad y busnes yn ymddangos yn addawol. 
Roedd ei sgiliau gwyddbwyll wedi gwella. 
Roedd wedi datblygu ei ddoniau gemau bwrdd. 
Gwisgodd ei ddillad yn frysiog a charlamu i lawr y grisiau. 
Rhoddodd ei got amdano ac yna dringodd yr ysgol yn bwyllog."""
sents = sents_str.split("\n")

docs = []
for sent in sents:
    doc = nlp(sent)
    docs.append(doc)
random.shuffle(docs)

for doc in docs:
    similarities = Counter()
    for another_doc in docs:
        if another_doc != doc:
            similarities[another_doc] = doc.similarity(another_doc)
    if similarities:
        most_similar, similarity = similarities.most_common(1)[0]
        print (doc, most_similar, similarity)



stopword_filtered_docs = []
for sent in sents:
    doc = nlp(sent, disable=["tagger","ner","parser"])
    #filtered_text = "".join([t.text_with_ws for t in doc if t.is_stop is False])
    filtered_text = " ".join([t.text for t in doc if t.is_stop is False])
    filtered_doc = nlp(filtered_text)
    stopword_filtered_docs.append(filtered_doc)

for doc in stopword_filtered_docs:
    similarities = Counter()
    for another_doc in stopword_filtered_docs:
        if another_doc != doc:
            similarities[another_doc] = doc.similarity(another_doc)
    if similarities:
        most_similar, similarity = similarities.most_common(1)[0]
        print (doc, most_similar, similarity)

