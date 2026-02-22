import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


nlp = spacy.load("fr_core_news_sm")

def analyze_sentences(sentences):
    """
    给每个sentence添加完整tokens信息
    """
    texts = [s["text"] for s in sentences]
    docs = list(nlp.pipe(texts))
    vectors = embedding_model.encode(texts)
    
    for sentence, doc, vector in zip(sentences, docs, vectors):
        tokens = []
        for token in doc:
            if token.is_space:
                continue

            tokens.append({
                "text": token.text,
                "lemma": token.lemma_,#原型
                "pos": token.pos_, #粗粒度词性标注            
                "tag": token.tag_,  #细粒度词性标注             
                "morph": token.morph.to_dict(),  #提取形态特征，返回词汇的数（单复数）、性（阴阳性）、时态等属性
                "start_char": token.idx,#起始字符索引
                "end_char": token.idx + len(token.text),#结束字符索引
                "is_punct": token.is_punct#是否为标点符号
            })

        sentence["tokens"] = tokens
        sentence["vector"] = vector.tolist()
    matrix = np.array([s["vector"] for s in sentences])
    similarities = matrix @ matrix.T
    for i, sentence in enumerate(sentences):
        sim_row = similarities[i]
        sim_row[i] = -1  # 排除自己
        most_similar_index = int(np.argmax(sim_row))
        sentence["most_similar"] = most_similar_index
    return sentences    

