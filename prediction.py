import torch
from model import GPTModel
from tokenizer import Tokenizer

def generate(model, tokenizer, text, max_length, device):
    #文本的输入一定要经过tokenizer的encode转为编码！
    input, att_mask = tokenizer.encode(text)
    '''将形状为 (seq_len,) 的张量转换为 (1, seq_len)，使其符合模型的输入格式（通常是批量输入）'''
    input = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)#在第0维度加一维
    stop = False
    input_len = len(input[0])
    while not stop:  # 使用 stop 变量控制循环
        if len(input[0]) - input_len > max_length:#当前序列长度-初始序列长度大于最大长度，则停止生成
            next_symbol = tokenizer.sep_token#如果大于了，设置分隔符并cat到末尾
            input = torch.cat(
                [input.detach(), torch.tensor([[next_symbol]], dtype=input.dtype, device=device)], -1)
            break
        '''模型的输出是一个词概率的分布！squeeze？'''
        projected, self_attns = model(input)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]#拿到最后一个维度，即词表的最大值（概率）
        next_word = prob.data[-1]#取 prob 的最后一个元素，即序列中最后一个位置预测的最可能的词
        next_symbol = next_word
        if next_symbol == tokenizer.sep_token:
            stop = True
        input = torch.cat(
            [input.detach(), torch.tensor([[next_symbol]], dtype=input.dtype, device=device)], -1)#不断拼接前面生成的字符，流式输出
    decode = tokenizer.decode(input[0].tolist())
    decode = decode[len(text):]#从解码后的字符串中去掉原始输入文本的部分，只保留模型生成的部分
    return "".join(decode)


def main():
  model_path="output/best.pt"
  vocab_path="data/vocab.json"
  max_length=128
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  tokenizer=Tokenizer(vocab_path)
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
  model.load_state_dict(torch.load(model_path))
  model.to(device)

  while(True):
    text=input("请输入：")
    if not text:
      continue
    if text=="q":
      break
    res=generate()
    print("AI: ",res)


if __name__=="__main__":
  main()