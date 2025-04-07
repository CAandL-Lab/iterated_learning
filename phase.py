import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 13})

# Dataset Hyper-parameters
n1 = 3 #n1 num sys inputs
n2 = 2 #n2 num sys outputs
k1 = 3 #k1 num nonsys reps input
k2 = 1  #k2 num nonsys reps output
r = 2 #r scale

# Training Hyper-parameters
lambdar = 0.001
tau = 1/((2**n1)*lambdar)
a_init = 8e-7
num_epochs = 1500

# Settings for Plot
percentage = True # Show norms as percentage of final norm (True) or actual norms (False)
trainings = ['dense', 'shallow', 'split'] # Makes sure a dense and shallow network are plotted together

# Deep and Shallow network dynamic formulas
def deep_dynamics(a, s, t):
	taus = (np.arange(t)/tau)
	exp_s = np.exp(-2*s*taus)
	return (a/(1-(1-(a/a_init))*exp_s ))

def shallow_dynamics(a, d, t):
	taus = (np.arange(t)/tau)
	exp_d = np.exp(-d*taus)
	return a*(1 - exp_d) + a_init*exp_d

# SV formulas
s1 = ( ((k1*r**2+2**n1)*(k2*r**2+2**n1))/(2**(2*n1)))**0.5
s2 = ( ((k1*r**2+2**n1)*(k2*r**2))/(2**(2*n1)))**0.5
s3 = ( (k1*k2*r**4)/(2**(2*n1)) )**0.5
d1 = (k1*r**2+2**n1)/(2**n1)
d2 = (k1*r**2)/(2**n1)
a1 = ((k2*r**2+2**n1)/(k1*r**2+2**n1))**0.5
a2 = ((k2*r**2)/(k1*r**2+2**n1))**0.5
a3 = (k2/k1)**0.5

# Gets SV trajectories
def plot_rel_norms(typer):
    if typer == 'dense' or typer == 'split':
        a1_traj = deep_dynamics(a1, s1, num_epochs)
        a2_traj = deep_dynamics(a2, s2, num_epochs)
        a3_traj = deep_dynamics(a3, s3, num_epochs)
    else:
        a1_traj = shallow_dynamics(a1, d1, num_epochs)
        a2_traj = shallow_dynamics(a2, d1, num_epochs)
        a3_traj = shallow_dynamics(a3, d2, num_epochs)
    if typer == 'dense' or typer == 'shallow':
        predicted_sys_norm = ( (n2*2**n1)/(k2*r**2+2**n1)*a1_traj**2)**0.5
        predicted_non_sys_norm = ( (k2*n2*r**2/(k2*r**2+2**n1))*a1_traj**2 \
                                  + (k2*(n1-n2)*r**2/(k2*r**2))*a2_traj**2 \
                                  + (2**n1-n1)*a3_traj**2 )**0.5
    else:
        predicted_sys_norm = ( (n2*2**n1)/(k2*r**2+2**n1)*a1_traj**2)**0.5
        predicted_non_sys_norm = ( (n1-n2)*a2_traj**2 + (2**n1-n1)*a3_traj**2 )**0.5

    if percentage:
        predicted_sys_norm = predicted_sys_norm/predicted_sys_norm[-1]
        predicted_non_sys_norm = predicted_non_sys_norm/predicted_non_sys_norm[-1]
    return predicted_sys_norm, predicted_non_sys_norm

# Plot relative norms for both the deep and shallow network
for typer in trainings:
    predicted_sys_norm, predicted_non_sys_norm = plot_rel_norms(typer)
    if typer == 'dense':
        plt.plot(predicted_sys_norm, predicted_non_sys_norm,label="Deep Network")
        plt.scatter(predicted_sys_norm[::50], predicted_non_sys_norm[::50],label="Deep Network 50 Epochs")
    elif typer == 'shallow':
        plt.plot(predicted_sys_norm, predicted_non_sys_norm,label="Shallow Network")
        plt.scatter(predicted_sys_norm[::50], predicted_non_sys_norm[::50],label="Shallow Network 50 Epochs")
    else:
        plt.plot(predicted_sys_norm, predicted_non_sys_norm,label="Deep Split Network")
        plt.scatter(predicted_sys_norm[::50], predicted_non_sys_norm[::50],label="Deep Split Network 50 Epochs")

# Add titles, etc to the plot and save
if percentage:
    plt.xlabel("Systematic Norm Percentage Progress")
    plt.ylabel("Non-Systematic Norm Percentage Progress")
else:
    plt.xlabel("Systematic Norm Progress")
    plt.ylabel("Non-Systematic Norm Progress")
#partition_type = ["Input", "Output"]
#input_norm_type = ["", "Input Normalized, "]
#title = "Relative Progress of {0}{1} Partitioned Norms".format(input_norm_type[remove_input], partition_type[partition_output])
plt.legend()
plt.grid()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
#plt.title(title)
plt.savefig("relative_norms.pdf")
plt.savefig("relative_norms.png", dpi=400)
