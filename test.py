# %%
import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.model import LanguageIdentificationModel

# %%
# df = pd.read_csv('./test.csv')
# """
# df.head()
#             file  label
# 0  test_0001.npy    NaN
# 1  test_0002.npy    NaN
# 2  test_0003.npy    NaN
# 3  test_0004.npy    NaN
# 4  test_0005.npy    NaN
# """

# %%
model = LanguageIdentificationModel()
model.load_state_dict(torch.load('./model/LanguageIdentificationModel_005.pt'))

# %%
test_path = glob.glob("./data/test_data/*.npy")
test_path.sort()

test_results = []
for path in tqdm(test_path):
    mel_spectrogram = np.load(path)
    if mel_spectrogram.shape[0] < 800:
        mel_spectrogram = np.concatenate((mel_spectrogram, np.zeros(
            (800 - mel_spectrogram.shape[0], 80))), axis=0)
    else:
        mel_spectrogram = mel_spectrogram[:800, :]
    mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

    mel_spectrogram = mel_spectrogram.unsqueeze(0)
    mel_spectrogram = mel_spectrogram.unsqueeze(0)
    outputs = model(mel_spectrogram)

    _, preds = torch.max(outputs, 1)

    test_results.append(
        [os.path.basename(path), preds.item()]
    )

# %%
test_results = pd.DataFrame(test_results, columns=['file', 'label'])
test_results.to_csv('./test_005.csv', index=False)
