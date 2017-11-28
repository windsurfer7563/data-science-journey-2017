import json
import semantics as sem
import re
from nltk import sent_tokenize
import gensim
from multiprocessing import Pool
import tqdm

word_with_dots = {'тыс','г','кг', 'м','А','Б','В','Г','Д','Е','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Э','Ю','Я'}

accentuations = {'Á':'А', 'á':'а', 'Ó':'О', 'ó':'о', 'É':'Е', 'é':'е', 'ý':'у', 'и́':'и','ы́':'ы', 'э́':'э', 'ю́':'ю', 'я́':'я'}


def uniq_words(text):
    return set(re.findall("\w+", text))


def read_data_model(file_name: str) -> dict:
    file = open(file_name, mode='r', encoding='utf-8')
    return json.load(file)


def write_data_model(file_name: str, data_model: dict):
    file = open(file_name, mode='w', encoding='utf-8')
    json.dump(data_model, file, separators=(',', ':'), ensure_ascii=False)





def make_bags(texts: list) -> list:
    bags = []
    for txt in texts:
        words = sem.canonize_words(uniq_words(txt))
        bags.append(words)
    return bags


def read_paragraphs(data) -> dict:
    paragraphs={}
    questions = {}
    print(data.shape)

    for idx, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
        p_id = row["paragraph_id"]

        if p_id not in paragraphs:
            text = row['paragraph']
            for word in word_with_dots:
                text = text.replace(word + ". ", word + "_ ")

            for k,v in accentuations.items():
                text = text.replace(k, v)

            sentences  = sent_tokenize(text)
            bags = make_bags(sentences)

            paragraphs[p_id] = {'sentences' : sentences,
                              'bags':  bags,
                              }

        q_id = row["question_id"]
        if q_id not in questions:
            text = row['question']

            #for word in word_with_dots:
            #    text = text.replace(word + ". ", word + " ")

            for k,v in accentuations.items():
                text = text.replace(k, v)

            sentences_q = [text]
            bags_q = make_bags(sentences_q)
            questions[q_id] = {'sentences' : sentences_q,
                              'bags':  bags_q,
                              }

    return paragraphs, questions


def read_paragraphs_multi(data, workers = None):
    workers = Pool(workers)
    paragraphs = workers.map_async(read_paragraphs, data)
    #workers.close()
    workers.join()
    return paragraphs.get()


def custom_w2v_model(paragraphs, questions):
    sentences = []
    for p in paragraphs.values():
        for b in p['bags']:
            sentences.append(b)
    for q in questions.values():
        for q in q['bags']:
            sentences.append(b)


    return gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4, hs=1, negative=0)
