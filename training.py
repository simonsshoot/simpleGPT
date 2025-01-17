import torch
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from model import GPTModel
from dataset import QADataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os,time,json,sys

def train_model(model,train_loader,val_loader,optimizer,criterion,device,num_epochs,model_output_dir,writer):
  batch_step=0
  best_val_loss=float('inf')    
  for epoch in range(num_epochs):
    time1=time.time()
    model.train()
    #tqdm在循环中显示进度条，desc设置显示的文字
    
    for index,data in enumerate(tqdm(train_loader,file=sys.stdout,desc="train epoch: "+str(epoch))):
      inputs_ids=data['input_ids'].to(device,dtype=torch.long)
      attention_mask=data['attention_mask'].to(device,dtype=torch.long)
      labels=data['labels'].to(device,dtype=torch.long)
      optimizer.zero_grad()
      outputs,dec_self_attns=model(inputs_ids,attention_mask)
      loss=criterion(outputs,labels.view(-1))
      loss.backward()
      #梯度裁剪
      torch.nn.utils.clip_grad_norm_(model.parameters(),1)
      optimizer.step()
      writer.add_scalar("loss/train",loss.item(),batch_step)
      batch_step+=1
      if index%100==0 or index==len(train_loader)-1:
        time2=time.time()
        tqdm.write(
          f"{index},epoch:{epoch}-loss:{str(loss)};lr:{optimizer.param_groups[0]['lr']};eachstep'stimespent:{(str(float(time2-time1)/float(index+0.0001)))}")
        
    #验证
    model.eval()
    val_loss=validate_model(model,criterion,device,val_loader)
    writer.add_scalar("loss/val",val_loss,epoch)
    print(f"val_loss:{val_loss} , epoch:{epoch}")
    #保存最优模型
    if val_loss<best_val_loss:
      best_val_loss=val_loss
      best_model_path=os.path.join(model_output_dir,"best.pt")
      os.makedirs(model_output_dir, exist_ok=True)
      print("save best model to ",best_model_path,"epoch: ",epoch)
      torch.save(model.state_dict(),best_model_path)

    #保存当前模型
    last_model_path=os.path.join(model_output_dir,"last.pt")
    os.makedirs(model_output_dir, exist_ok=True)
    print("save last model to ",last_model_path,"epoch: ",epoch)
    torch.save(model.state_dict(),last_model_path)

def validate_model(model,criterion,device,val_loader):
  running_loss=0.0
  with torch.no_grad():
    for index,data in enumerate(tqdm(val_loader,file=sys.stdout,desc="validation data")):
      inputs_ids=data['input_ids'].to(device,dtype=torch.long)
      attention_mask=data['attention_mask'].to(device,dtype=torch.long)
      labels=data['labels'].to(device,dtype=torch.long)
      outputs,dec_self_attns=model(inputs_ids,attention_mask)
      loss=criterion(outputs,labels.view(-1))
      running_loss+=loss.item()
    val_loss=running_loss/len(val_loader)
    return val_loss

def main():
  train_json_path="data/train.jsonl"
  val_json_path="data/val.jsonl"
  vocab_path="data/vocab.json"
  max_length=120 #最大长度
  epochs=15
  batch_size=128
  learning_rate=0.25*1e-4
  model_output_dir="output"
  logs_dir="logs"
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  tokenizer=Tokenizer(vocab_path)
  #模型参数
  model_param={
    "d_model":768,#嵌入层大小
    "d_ff":2048,#FFN层大小
    "d_k":64,#k的大小
    "d_v":64,
    "n_layers":6,#解码层数量
    "n_heads":8,
    "max_pos":1800,#位置编码的长度
    "device":device,
    "vocab_size":tokenizer.get_vocab_size()
  }
  model=GPTModel(**model_param)
  print("start load training data...")
  train_params={
    "batch_size":8,
    "shuffle":True,
    "num_workers":4,
  }
  training_set=QADataset(train_json_path,tokenizer,max_length)
  training_loader=DataLoader(training_set,**train_params)
  print(training_loader)
  print("start load validation data...")
  val_params={
    "batch_size":8,
    "shuffle":False,
    "num_workers":4,
  }
  val_set=QADataset(val_json_path,tokenizer,max_length)
  val_loader=DataLoader(val_set,**val_params)
  #日志记录
  writer=SummaryWriter(logs_dir)

  optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
  criterion=torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
  model=model.to(device)
  #开始训练
  print("start training...")
  train_model(
    model=model,
    train_loader=training_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=epochs,
    model_output_dir=model_output_dir,
    writer=writer
  )
  writer.close()


if __name__=="__main__":
  main()