import pymorphy2
import gensim
import logging
import numpy as np
import functools

# http://ling.go.mail.ru/static/models/ruscorpora.model.bin.gz
WORD2VEC_MODEL_FILE = 'data/ruwikiruscorpora_0_300_20.bin.gz'

morph_analyzer = pymorphy2.MorphAnalyzer()

grammar_map_MY_STEM = {
    'NOUN': '_S',
    'VERB': '_V', 'INFN': '_V', 'GRND': '_V', 'PRTF': '_V', 'PRTS': '_V',
    'ADJF': '_A', 'ADJS': '_A',
    'ADVB': '_ADV',
    'PRED': '_PRAEDIC'
}

grammar_map_POS_TAGS =  {
    'NOUN': '_NOUN',
    'VERB': '_VERB', 'INFN': '_VERB', 'GRND': '_VERB', 'PRTF': '_VERB', 'PRTS': '_VERB',
    'ADJF': '_ADJ', 'ADJS': '_ADJ',
    'ADVB': '_ADV',
    'PRED': '_ADP'
}


# http://textmechanic.com/text-tools/basic-text-tools/remove-duplicate-lines/
# http://www.codeisart.ru/blog/python-shingles-algorithm/
def canonize_words(words: list, grammar_map: dict = grammar_map_POS_TAGS) -> list:
    stop_words = ('назовите','быть', 'какой','как','какова','каков','мой', 'наш', 'ваш', 'их', 'его', 'её', 'их',
                  'этот', 'тот', 'где', 'который','такой', 'либо', 'нибудь', 'нет', 'да')

    normalized = []
    for w in words:
        forms = morph_analyzer.parse(w.lower())
        try:
            form = max(forms, key=lambda x: (x.score, x.methods_stack[0][2]))
        except Exception:
            form = forms[0]
            #print("Error finding form: {}".format(form))
        if not (form.tag.POS in ['CONJ', 'NPRO']
                #or 'Name' in form.tag
                #or 'UNKN' in form.tag
                or form.normal_form in stop_words):  # 'ADJF'
            norm_word = form.normal_form.replace("ё", "е")
            normalized.append(norm_word + grammar_map.get(form.tag.POS, ''))
    return normalized





def load_w2v_model(file_name: str):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True, encoding='utf-8')
    print("word2vec model '%s' loaded" % file_name)
    return w2v_model


def semantic_density(bag: list, w2v_model, unknown_coef=0.0) -> float:
    sim_sum = 0.0
    divisor = 0
    # weight_sum = 0.0
    for i in range(len(bag)):
        for j in range(i + 1, len(bag)):
            if bag[i] != bag[j]:
                divisor += 1
                # weight = 1 / (j - i)
                # weight_sum += weight
                try:
                    # sim_sum += w2v_model.similarity(bag[i], bag[j]) # * weight
                    sim_sum += np.dot(w2v_model[bag[i]], w2v_model[bag[j]]) # vectors already normalized
                except:
                    sim_sum += unknown_coef # * weight
        return sim_sum / divisor if divisor > 0 else 0.0 # / weight_sum

@functools.lru_cache(maxsize=2 ** 19)
def bag_to_matrix(bag: list, w2v_model):
    mx = []
    for i in range(len(bag)):
        try:
            mx.append(w2v_model[bag[i]])
        except:
            pass
    return np.vstack(mx) if len(mx) > 0 else np.array([])


def most_similar(w2v_model, positive="", negative="", topn=10):
    pos_bag = canonize_words(positive.split())
    neg_bag = canonize_words(negative.split())
    return w2v_model.most_similar(pos_bag, neg_bag, topn) if len(positive) > 0 else ()


#@functools.lru_cache(maxsize=2 ** 19)
def semantic_similarity_fast(mx1, mx2) -> float:
    return np.sum(np.dot(mx1, mx2.T)) if len(mx1) > 0 and len(mx2) > 0 else 0.0


#@functools.lru_cache(maxsize=2 ** 19)
def semantic_similarity_fast_log(mx1, mx2) -> float:
    return np.sum(np.dot(mx1, mx2.T)) * np.log10(len(mx2)) / (len(mx2) * len(mx1)) \
           if len(mx1) > 0 and len(mx2) > 0 else 0.0

#@functools.lru_cache(maxsize=2 ** 19)
def semantic_similarity(bag1, bag2: list, w2v_model, unknown_coef=0.0) -> float:
    sim_sum = 0.0
    intersection = []
    simslen = 0
    for i in range(len(bag1)):
        for j in range(len(bag2)):
            try:
                sims = 0
                #sims = w2v_model.similarity(bag1[i], bag2[j])
                sim_sum += sims
                sim_sum += np.dot(w2v_model[bag1[i]], w2v_model[bag2[j]]) # vectors already normalized
                simslen += 1
                if sims > 0.6:
                    intersection.extend([bag1[i], bag2[j]])
                
            except:
                sim_sum += unknown_coef
    return sim_sum / (len(bag1) * len(bag2)) if simslen > 0 else 0.0, intersection
    


def semantic_association(bag: list, w2v_model, topn=10) -> list:
    positive_lst = [w for w in bag if w in w2v_model.vocab]
    if len(positive_lst) > 0:
        assoc_lst = w2v_model.most_similar(positive=positive_lst, topn=topn)
        return [a[0] for a in assoc_lst]
    else:
        print('empty association for bag:', bag)
        return ['пустота_S']







