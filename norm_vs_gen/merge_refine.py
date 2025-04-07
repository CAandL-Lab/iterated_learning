import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']="\\usepackage{amsmath}"

initial_norm = 1.5056499397977066 #1.3927149222519828 (3,1,3,1,1)
best_early_stop = 0.8534285902194283 #0.029934058915003465 (3,1,3,1,1)

# Plot Non-systematic Norm final value vs Generations
non_norms = np.insert(np.loadtxt('dense_norm_v_gen.txt', dtype=np.float64), 0, initial_norm)
plt.plot(non_norms, label='Deep Network', color='red')
plt.scatter(np.arange(non_norms.shape[0]), non_norms, color='red')
non_norms = np.insert(np.loadtxt('shallow_norm_v_gen.txt', dtype=np.float64), 0, initial_norm)
plt.plot(non_norms, label='Shallow Network', color='blue')
plt.scatter(np.arange(non_norms.shape[0]), non_norms, color='blue')
non_norms = np.insert(np.loadtxt('split_norm_v_gen.txt', dtype=np.float64), 0, initial_norm)
plt.ylabel(r"Final $\Gamma_y$-Norm")
plt.xlabel("Generation Number (Generation length of " + str(non_norms.shape[0]) + " epochs)")
plt.plot(non_norms, label='Split Network', color='green')
plt.scatter(np.arange(non_norms.shape[0]), non_norms, color='green')
plt.plot(np.ones(non_norms.shape[0])*best_early_stop, color = 'orange', label='Best Split Net Early Stop', linestyle='dashed')
#plt.title(r'$\Gamma_y$-Norm over Generations: Refinement Towards Systematicity')
plt.legend(loc='upper right', bbox_to_anchor=(0.52, 0.53, 0.5, 0.5))
plt.grid()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.savefig("norms_vs_gen.pdf")
plt.savefig("norms_vs_gen.png",dpi=400)
plt.close()
