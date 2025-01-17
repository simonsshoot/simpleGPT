import torch

from model import GPTModel

def main():
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #模型参数
  model_params={
    "d_model":768,#嵌入层大小
    "d_ff":2048,#FFN层大小
    "d_k":64,#k的大小
    "d_v":64,
    "n_layers":6,#解码层数量
    "n_heads":8,
    "max_pos":1800,#位置编码的长度
    "device":device,
    "vocab_size":48256#词表大小
  }

  '''经典传参方式！字典解包传播'''
  model=GPTModel(**model_params)
  #p.numel() 计算每个参数的元素数量
  total_params=sum(p.numel() for p in model.parameters())
  print(model)
  print(f"Total number of parameters: {total_params}")

if __name__=="__main__":
  main()


'''
GPTModel(
  (decoder): Decoder(
    (embedding): Embedding(1800, 768)
    (pos_encoding): PositionalEncoding(
      (pos_embedding): Embedding(48256, 768)
    )
    (layers): ModuleList(
      (0-5): 6 x DecoderLayer(
        (attention): MultiHeadAttention(
          (w_q): Linear(in_features=768, out_features=512, bias=False)
          (w_k): Linear(in_features=768, out_features=512, bias=False)
          (w_v): Linear(in_features=768, out_features=131072, bias=False)
          (fc): Linear(in_features=131072, out_features=768, bias=False)
          (layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (pos_ffn): PositionwiseFeedForwardNet(
          (fc): Sequential(
            (0): Linear(in_features=768, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=768, bias=True)
          )
          (layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (projection): Linear(in_features=768, out_features=48256, bias=True)
)
Total number of parameters: 1288843264
'''