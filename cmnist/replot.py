import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

### This script just replots the dense and split network performance together on the same plot ###

dense_sys_train_mean = np.mean(np.genfromtxt('dense/train_accs_sys.txt'), axis=0)
dense_non_train_mean = np.mean(np.genfromtxt('dense/train_accs_non.txt'), axis=0)
split_sys_train_mean = np.mean(np.genfromtxt('split/train_accs_sys.txt'), axis=0)
split_non_train_mean = np.mean(np.genfromtxt('split/train_accs_non.txt'), axis=0)
dense_sys_train_std = np.std(np.genfromtxt('dense/train_accs_sys.txt'), axis=0)
dense_non_train_std = np.std(np.genfromtxt('dense/train_accs_non.txt'), axis=0)
split_sys_train_std = np.std(np.genfromtxt('split/train_accs_sys.txt'), axis=0)
split_non_train_std = np.std(np.genfromtxt('split/train_accs_non.txt'), axis=0)

dense_sys_test_mean = np.mean(np.genfromtxt('dense/test_accs_sys.txt'), axis=0)
dense_non_test_mean = np.mean(np.genfromtxt('dense/test_accs_non.txt'), axis=0)
split_sys_test_mean = np.mean(np.genfromtxt('split/test_accs_sys.txt'), axis=0)
split_non_test_mean = np.mean(np.genfromtxt('split/test_accs_non.txt'), axis=0)
dense_sys_test_std = np.std(np.genfromtxt('dense/test_accs_sys.txt'), axis=0)
dense_non_test_std = np.std(np.genfromtxt('dense/test_accs_non.txt'), axis=0)
split_sys_test_std = np.std(np.genfromtxt('split/test_accs_sys.txt'), axis=0)
split_non_test_std = np.std(np.genfromtxt('split/test_accs_non.txt'), axis=0)

epochs = 301
_,_,bars = plt.errorbar(np.arange(epochs), dense_sys_train_mean, dense_sys_train_std, label='Dense Systematic Error')
[bar.set_alpha(0.5) for bar in bars]
_,_,bars = plt.errorbar(np.arange(epochs), dense_non_train_mean, dense_non_train_std, label='Dense Non-systematic Error')
[bar.set_alpha(0.5) for bar in bars]
_,_,bars = plt.errorbar(np.arange(epochs), split_sys_train_mean, split_sys_train_std, label='Split Systematic Error')
[bar.set_alpha(0.5) for bar in bars]
_,_,bars = plt.errorbar(np.arange(epochs), split_non_train_mean, split_non_train_std, label='Split Non-systematic Error')
[bar.set_alpha(0.5) for bar in bars]
#plt.title('Train Normalized Error')
plt.xlabel('Epoch Number')
plt.ylabel('Normalized Error')
plt.legend()
plt.grid()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.savefig('cmnist_train.png')
plt.close()

_,_,bars = plt.errorbar(np.arange(epochs), dense_sys_test_mean, dense_sys_test_std, label='Dense Systematic Error')
[bar.set_alpha(0.5) for bar in bars]
_,_,bars = plt.errorbar(np.arange(epochs), dense_non_test_mean, dense_non_test_std, label='Dense Non-systematic Error')
[bar.set_alpha(0.5) for bar in bars]
_,_,bars = plt.errorbar(np.arange(epochs), split_sys_test_mean, split_sys_test_std, label='Split Systematic Error')
[bar.set_alpha(0.5) for bar in bars]
_,_,bars = plt.errorbar(np.arange(epochs), split_non_test_mean, split_non_test_std, label='Split Non-systematic Error')
[bar.set_alpha(0.5) for bar in bars]
#plt.title('Test Normalized Error')
plt.xlabel('Epoch Number')
plt.ylabel('Normalized Error')
plt.legend()
plt.grid()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.savefig('cmnist_test.png')
