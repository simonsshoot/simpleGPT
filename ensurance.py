import json
from tokenizer import Tokenizer
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']#中文文本黑体显示

def get_num_tokens(file_path,tokenizer):
  input_num_tokens = []
  with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
      line=json.loads(line)
      question=line['question']
      answer=line['answer']
      tokens,att_mask=tokenizer.encode(question,answer)
      input_num_tokens.append(len(tokens))
    return input_num_tokens

#统计 num_tokens 列表中数值落在不同区间内的数量
def count_intervals(num_tokens,interval):
  max_value=max(num_tokens)
  intervals_count={}
  for lower_bound in range(0,max_value+1,interval):
    upper_bound=lower_bound+interval
    count=len([num for num in num_tokens if num>=lower_bound and num<upper_bound])
    intervals_count[f"{lower_bound}--{upper_bound}"]=count

  return intervals_count

def main():
  train_data_path='data/train.jsonl'
  tokenizer=Tokenizer("data/vocab.json")
  input_num_tokens=get_num_tokens(train_data_path,tokenizer)
  intervals_count=count_intervals(input_num_tokens,20)
  print(intervals_count)
  x=[k for k in intervals_count.keys()]
  y=[v for v in intervals_count.values()]
  plt.figure(figsize=(11,8))#创建画布
  bars=plt.bar(x,y)#绘制柱状图，返回每个柱的对象
  plt.title('训练集token的分布情况')
  plt.ylabel('数量')
  plt.xticks(rotation=45)
  for bar in bars:
    yval=bar.get_height()
    #plt.text()：在每个柱的顶部中心位置显示其数值。x坐标，y坐标，显示的内容，位置
    plt.text(bar.get_x()+bar.get_width()/2,yval,int(yval),ha='center')
  plt.show()

if __name__ == '__main__':
  main()
