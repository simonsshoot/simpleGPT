'''构建词表，这里我将一个字作为一个词，也可以优化通过分词器分词后的词构建词表，需要注意的时，词表需要拼接三个特殊Token，用于表示特殊意义： pad 占位、unk 未知、sep 结束'''

import json

def build_vocab(file_path):
  texts=[]
  with open(file_path, 'r', encoding='utf-8') as r:
    for line in r:
      if not line:#跳过空行
        continue
      try:
        line=json.loads(line)#每一行解析为json对象，即python字典
      except:
        print(f"json解析失败:{line}")
        exit()
      question=line['question'] 
      answer=line['answer']
      texts.append(question)
      texts.append(answer)

  #拆分token
  words=set()#确保不重复
  for t in texts:
    if not t:
      continue
    for word in t.strip():#去除字符串 开头和结尾 的空白字符
      words.add(word)
  words=list(words)
  words.sort()

  word2id={"<pad>":0,"<unk>":1,"<sep>":2}
  #构建词表
  word2id.update({word:i+len(word2id) for i,word in enumerate(words)})#从2开始往后添加
  id2word=list(word2id.keys())
  vocab={"word2id":word2id,"id2word":id2word}#双向映射便于查找
  vocab=json.dumps(vocab,ensure_ascii=False)#序列化为json字符串
  with open("data/vocab.json","w",encoding="utf-8") as w:
    w.write(vocab)
  
  print("词表构建完成，共{}个词".format(len(word2id)))

if __name__=="__main__":
  file_path="data/rawtrain.jsonl"
  build_vocab(file_path)


