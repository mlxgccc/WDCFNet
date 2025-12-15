from thop import profile
import torch
import time
from net.WDCFNet import WDCFNet

device = 'cuda'

model = WDCFNet().to(device)
model.eval()

# 三个输入张量
input1 = torch.rand(1, 3, 256, 256).to(device)  # x
input2 = torch.rand(1, 3, 256, 256).to(device)  # x1
input3 = torch.rand(1, 3, 256, 256).to(device)  # x2

batch = (input1, input2, input3)

# --------- Time ---------
torch.cuda.synchronize()
t0 = time.time()
_ = model(batch)
torch.cuda.synchronize()
print("Time:", time.time() - t0)

# --------- Params ---------
n_param = sum(p.nelement() for p in model.parameters())
print(f"Params: {n_param/2**20:.3f} M")

# --------- FLOPs / MACs ---------
macs, params = profile(model, inputs=(batch,))
print(f"FLOPs: {macs/(2**30):.3f} G")
#训练时可去掉LCA的多余残差链接，不生效。 增加计算量 但用训练好的模型测评时要加上。