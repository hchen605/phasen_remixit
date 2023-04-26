
import os
import numpy as np
import pandas as pd


csv_path = '../output/phasen_vb/enhanced/_results.csv'
test_noisy_dir = ['/home/koredata/hsinhung/speech/vb_demand/noisy_testset_wav']
train_clean_dir = ['/home/koredata/hsinhung/speech/vb_demand/clean_trainset_28spk_wav']
train_noisy_dir = ['/home/koredata/hsinhung/speech/vb_demand/noisy_trainset_28spk_wav']
test_clean_dir = ['/home/koredata/hsinhung/speech/vb_demand/clean_testset_wav']

f_test = open("test_vb_pesq_below_1p5.txt", "w")
data_df = pd.read_csv(csv_path, sep=',')   
wavpath = data_df['filename'].tolist()
pesq = data_df['noisy_pesq'].to_list()
#print(pesq)
mask = np.array(pesq) < 1.5
#print(mask)
wavpath = np.array(wavpath)
wav_mask = wavpath[mask]
#print(wav_mask)

for path in test_noisy_dir:
    for (dirpath, dirnames, filenames) in os.walk(path):
        print(dirpath)
        #print(len(filenames))
        for f in filenames:
            if not f.endswith((".WAV", ".wav")) or f[:-4] not in wav_mask:
                continue

            clean_path = dirpath
            clean_path = clean_path.replace("noisy", "clean")
            
            noisy_path = os.path.join(dirpath, f)
            clean_path = os.path.join(clean_path, f)
            
            f_test.write(noisy_path)
            f_test.write(' ')
            f_test.write(clean_path)
            f_test.write('\n')
          
