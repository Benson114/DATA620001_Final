# %%
from torchviz import make_dot
from src.dataloader import *
from src.model import LanguageIdentificationModel

import matplotlib.pyplot as plt
import librosa.display

mel_spectrogram, labels = load_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LanguageIdentificationModel()
model = model.to(device)

# %%
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram[0], y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram example')
plt.tight_layout()
plt.savefig("./img/mel_spectrogram_example.png", dpi=300)
plt.show()

# %%
x = [mel_spectrogram[i].shape[0] for i in range(len(mel_spectrogram))]
plt.figure(figsize=(10, 4))
plt.hist(x, bins=100)
plt.title('Mel spectrogram length distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("./img/mel_spectrogram_length.png", dpi=300)
plt.show()

# %%
input_tensor = torch.rand(1, 1, 800, 80).to(device)
output = model(input_tensor)
if isinstance(output, tuple):
    output = output[0]
vis_graph = make_dot(output, params=dict(model.named_parameters()))
vis_graph.render("model_visualization")


# 这块有BUG，跑不了
# 使用 torchsummary 展示模型概览
# from torchsummary import summary
# summary(model, (1, 800, 80))
