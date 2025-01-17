import os.path
#数据集分割为训练集和验证集
def seplit_dataset(file_path,output_path):
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  datas=[]
  with open(file_path,'r',encoding='utf-8') as f:
    for line in f:
      if not line or line=='':
        continue
      datas.append(line)
  train = datas[0:10000]
  val=datas[10000:11000]
  with open(os.path.join(output_path,'train.jsonl'),'w',encoding='utf-8') as f:
    for line in train:
      f.write(line)
      f.flush()#确保每一条数据都被及时写入到磁盘
  
  with open(os.path.join(output_path,'val.jsonl'),'w',encoding='utf-8') as f:
    for line in val:
      f.write(line)
      f.flush()
  print("train count: ",len(train))
  print("val count: ",len(val))

if __name__ == '__main__':
  file_path="data/rawtrain.jsonl"
  seplit_dataset(file_path,output_path='data')