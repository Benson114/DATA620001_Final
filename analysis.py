# %%
from src.dataloader import *

mel_spectrogram, labels = load_data()

# %%
# 绘制一个mel频谱图
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram[0], y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram example')
plt.tight_layout()
# 将图像保存为png
plt.savefig("./img/mel_spectrogram_example.png", dpi=300)
plt.show()

# %%
x = [mel_spectrogram[i].shape[0] for i in range(len(mel_spectrogram))]
# 绘制x的分布直方图
plt.figure(figsize=(10, 4))
plt.hist(x, bins=100)
plt.title('Mel spectrogram length distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.tight_layout()
# 将图像保存为png
plt.savefig("./img/mel_spectrogram_length.png", dpi=300)
plt.show()

# %%
import torch
from torchsummary import summary
from src.model import LanguageIdentificationModel  # 确保 model.py 文件在同一目录下
from torchviz import make_dot

# 创建模型实例并将其移到合适的设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LanguageIdentificationModel()
model = model.to(device)

# 使用 torchsummary 展示模型概览
summary(model, (1, 800, 80))

# 生成一个随机的输入张量，并将其移到与模型相同的设备上
input_tensor = torch.rand(1, 1, 800, 80).to(device)

# 使用模型进行前向传播
output = model(input_tensor)

# 打印输出类型
print("Output type:", type(output))

# 如果输出是一个元组，我们只取第一个元素来可视化
if isinstance(output, tuple):
    output = output[0]

# 使用 torchviz 生成模型结构图
vis_graph = make_dot(output, params=dict(model.named_parameters()))

# 保存可视化图像
vis_graph.render("model_visualization")

# %%
# 生成该项目的文件树的命令行输出
#
