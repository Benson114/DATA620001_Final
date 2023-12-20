# %%
# 导入模块
from src.dataloader import *
from src.model import *
from src.train import *

# %%
# 读取数据集
mel_spectrogram, labels = load_data()
uniform(mel_spectrogram)
load_kwargs = {
    'valid_size': 0.2,
    'batch_size': 128,
    'random_state': 42
}
train_loader, valid_loader = split_data(mel_spectrogram, labels, **load_kwargs)
print("数据集加载完成.")

# %%
# 定义模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LanguageIdentificationModel()
model = model.to(device)
print("模型将在{}上训练.".format(device))

# %%
# 训练模型
fit_kwargs = {
    'epochs': 15,
    'learning_rate': 0.001,
    'device': device
}
fit(model, train_loader, valid_loader, **fit_kwargs)

# %%
# 保存模型
current_idx = "005"
torch.save(model.state_dict(), './model/LanguageIdentificationModel_{}.pt'.format(current_idx))
print("模型已保存至./model/LanguageIdentificationModel_{}.pt.".format(current_idx))
