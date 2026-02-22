import spacy
nlp = spacy.load("fr_core_news_sm")


def analyze_sentences(sentences):
    """
    给每个sentence添加完整tokens信息
    """
    texts = [s["text"] for s in sentences]
    docs = list(nlp.pipe(texts))
    for sentence, doc in zip(sentences, docs):
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

    return sentences

