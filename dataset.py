#用于创建训练模型需要的数据集
from torch.utils.data import Dataset
import torch
import json
import numpy as np

class QADataset(Dataset):
  def __init__(self, data_path, tokenizer,max_length)->None:#箭头用于函数的类型注解，表示函数返回类型
    super().__init__()
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.data=[]
    if data_path:
      with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:  
          if not line or line=="":
            continue
          json_line = json.loads(line)
          question = json_line['question']
          answer = json_line['answer']
          #每一个问题和回答对都处理为一个字典
          self.data.append(
            {"question": question, "answer": answer}
          )
    print("data load , size: ",len(self.data))

  #用于将输入的 question 和 answer 文本预处理为模型可以接受的输入格式
  def preprocess(self,question,answer):
    #拿到token ID 序列和注意力掩码
    encode,att_mask = self.tokenizer.encode(question,answer,max_length=self.max_length,pad_to_max_length=True)
    '''
    input_ids：去掉最后一个 token 的 token ID 序列，作为模型的输入。
    att_mask：去掉最后一个 token 的注意力掩码，与 input_ids 对应。
    labels：去掉第一个 token 的 token ID 序列，作为模型的标签（目标输出）
    为什么这么做？——自回归模型
    input_ids 去掉最后一个 token：是因为模型的输入序列是用来预测下一个 token，而不包括预测的最后一个 token。
    labels 去掉第一个 token：是因为目标输出是模型需要预测的下一个 token，因此目标序列应该从第一个 token 的下一个开始。

    错位的感觉
    '''
    input_ids = encode[:-1]
    att_mask=att_mask[:-1]
    labels = encode[1:]
    return input_ids,att_mask,labels
  
  def __getitem__(self, index):
    item_data=self.data[index]
    input_ids,att_mask,labels=self.preprocess(**item_data)
    return {
      "input_ids": torch.LongTensor(np.array(input_ids)),
      "attention_mask": torch.LongTensor(np.array(att_mask)),
      "labels": torch.LongTensor(np.array(labels))
    }
  
  def __len__(self):
    return len(self.data)
