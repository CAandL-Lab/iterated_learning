import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad
import jax.numpy as jnp

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 13})
plt.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']="\\usepackage{amsmath}"

def svs(A, U, VT, num_svds, k2):
    # Gets singular values of A using A's singular vectors U and VT (keeps singular values aligned for plotting, unlike np.linalg.svd)
    S = np.dot(U.T, np.dot(A,VT.T))
    small_length = np.min([S.shape[0], S.shape[1]])
    s = np.array([S[i,i] for i in range(small_length)])
    if k2 == 0:
        s = np.sort(s)[::-1]
    s = s[:num_svds]
    return U, s, VT

def gen_binary_patterns(num_features):
    # Generates every unique binary digit possible with num_feature bits
    data = np.ones((2**num_features, num_features))*-1.0
    for j in np.arange(0, num_features, 1):
        step = 2**(j+1)
        idx = [list(range(i,i+int(step/2))) for i in np.arange(int(step/2),2**num_features,step)]
        idx = np.concatenate(idx)
        data[idx,j] = 1
    data = np.flip(data, axis=1)
    return data

def dynamic_formula(n1, n2, k1, k2, r, a_init, num_time_steps, num_svds):
  # Returns predictions of the training dynamics of the linear net using the formulas for the SVs for the first generation of training.

  # Sets up some constant matrices consistent with those used in the appendix for the SVD equations
  A = np.dot(np.dot(Y[:n2], X[:n1].T).T, np.dot(Y[:n2], X[:n1].T))
  B = np.dot(np.dot(Y[:n2].T,Y[:n2]), X[:n1].T)
  C = np.dot(np.dot(X[:n1].T,X[:n1]),X[:n1].T)
  Ax = np.dot(np.dot(X[:n1], X[:n1].T).T, np.dot(X[:n1], X[:n1].T))

  if k1>0 and k2>0:
    XZ = (1/(k1*k2))**0.5*np.eye(2**n1) - (1/((k1*k2)**0.5*2**n1))*np.dot(X[:n1].T, X[:n1])
    T,H,P = np.linalg.svd(XZ)
    T = T[:, :2**n1-n1] 
    P = (1/(k1*k2))**0.5*P[:2**n1-n1]
  else:
    XZ = (1/(k1))**0.5*np.eye(2**n1) - (1/((k1)**0.5*2**n1))*np.dot(X[:n1].T, X[:n1])
    T,H,P = np.linalg.svd(XZ)
    T = T[:, :2**n1-n1]
    P = (1/(k1))**0.5*P[:2**n1-n1]

  # Predicting the input-output covariance and input covariance SV formulas and storing in diagonal-matrix form
  # to get correct order of SVs
  sv1 = ((k1*r**2+2**n1)*(k2*r**2+2**n1)/2**(6*n1))**0.5
  sv2 = ((k1*r**2+2**n1)*(k2*r**2)/2**(2*n1))**0.5
  sv3 = ((k1*k2)**0.5*r**2)/(2**n1)
  sxv1 = (k1*r**2+2**n1)/2**(n1)
  sxv3 = (k1*r**2)/(2**n1)

  sv_preds = sv1*A + sv2*(np.eye(n1) - (1/2**(2*n1))*A)
  sxv_preds_inv = sxv1*np.eye(n1)

  # Flattening SV formulas back into a list now that the correct order is obtained
  sxv_preds_inv = [1/sxv_preds_inv[i,i] for i in range(sxv_preds_inv.shape[0]) if sv_preds[i,i] > 0]
  sv_preds = [sv_preds[i,i] for i in range(sv_preds.shape[0]) if sv_preds[i,i] > 0]

  if k2 > 0: 
      for _ in range(2**n1 - n1): sv_preds.append(sv3)
      for _ in range(2**n1 - n1): sxv_preds_inv.append(1/sxv3)

  # Getting unique SV positions in the list (used to calculate the norm trajectories using unique SV trajectories)
  unique_svs = np.unique(sv_preds[:n1])[::-1]
  SV11_i = np.argmax(sv_preds == unique_svs[0])
  if n1 > n2 and n2 > 0 and k2 > 0:
      SV12_i = np.argmax(sv_preds == unique_svs[1])
  else:
      SV12_i = np.argmax(sv_preds == unique_svs[0])
  SV21_i = -1
  sv_indices = [SV11_i, SV12_i, SV21_i]

  # Predicting the left singular vectors. Used to get correctly ordered SVs from the simulated network mapping.
  U1 = (1/2**n1)**0.5*(1/(k2*r**2+2**n1))**0.5
  U2 = (1/2**(3*n1))**0.5*((r**2)/(k2*r**2+2**n1))**0.5
  if k2 > 0:
      U3 = (1/2**(3*n1))**0.5*((r**2)/(k2*r**2))**0.5
      U4 = (1/(k1*k2))**0.25
      U5 = (1/(k1*k2))**0.125*(1/2**(1*n1))

  U_preds = U1*np.dot(Y[:n2],X[:n1].T)
  for _ in range(k2):
      U_preds = np.vstack([U_preds, U2*B + U3*(C-B)])

  if k2 > 0:
      U_nonsys_part = np.zeros((n2, 2**n1 - n1))
      for _ in range(k2):
          U_nonsys_part = np.vstack([U_nonsys_part, T])
      U_preds = np.hstack([U_preds, U_nonsys_part])
      U_preds = np.hstack([U_preds, np.zeros((U_preds.shape[0], n2+(k2*2**n1) - (2**n1)))]) 

  # Predicting the transposed right singular vectors. Used to get correctly ordered SVs from the simulated network mapping.
  VT1 = (2**n1)**0.5*(1/(k1*r**2+2**n1))**0.5
  VT2 = (1/2**n1)**0.5*((r**2)/(k1*r**2+2**n1))**0.5

  V_T_preds = VT1*np.eye(n1)
  for _ in range(k1):
      V_T_preds = np.hstack([V_T_preds, VT2*X[:n1]])

  V_T_nonsys_part = np.zeros((2**n1 - n1, n1))
  for _ in range(k1):
      V_T_nonsys_part = np.hstack([V_T_nonsys_part, P])

  V_T_preds = np.vstack([V_T_preds, V_T_nonsys_part])
  V_T_preds = np.vstack([V_T_preds, np.zeros((n1+(k1*2**n1) - (2**n1), V_T_preds.shape[1]))])

  # Getting SVs in numpy arrays, setting up time indices array and exponential components of SV dynamics formula
  s = np.array(sv_preds)
  d_inv = np.array(sxv_preds_inv)
  taus = (np.arange(0,num_time_steps,1).reshape(num_time_steps,1)/tau)
  exp_s = np.exp(-2*s[:num_svds]*taus)

  # Predict SV Trajectories using dynamics formula
  numerator = s*d_inv
  denominator = 1 - (1 - (s*d_inv/a_init))*exp_s
  sv_trajectory_plots = np.zeros(denominator.shape)
  num_sig_svs = numerator[numerator > 0].shape[0]
  sv_trajectory_plots[:,:num_sig_svs] = numerator[:num_sig_svs].reshape(1, num_sig_svs)/denominator[:,:num_sig_svs]

  # Predict Frobenius Norm Trajectories using SV dynamics (partitioned by output only, ie: full input used for mappings) 
  predicted_sys_norm = ( (n2*2**n1)/(k2*r**2+2**n1)*sv_trajectory_plots[:,SV11_i]**2)**0.5
  
  if k2 > 0:
      predicted_non_norm = ( (k2*n2*r**2/(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + (n1-n2)*sv_trajectory_plots[:,SV12_i]**2 \
                               + (2**n1-n1)*sv_trajectory_plots[:,SV21_i]**2 )**0.5
  else:
      predicted_non_norm = np.zeros(predicted_sys_norm.shape)
      
  # Predict Frobenius Norm Trajectories using SV dynamics (partitioned by input and output)
  predicted_sys_sys_norm = ( (n2*2**(2*n1))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2)**0.5
  predicted_non_sys_norm = ( (n2*2**n1*(k1*r**2))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2)**0.5
  
  if k2 > 0:
      predicted_sys_non_norm = ( (k2*n2*r**2*2**n1)/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + ((n1-n2)*2**n1)/(k1*r**2+2**n1)*sv_trajectory_plots[:,SV12_i]**2 )**0.5
      predicted_non_non_norm = ( (k2*n2*r**2*(k1*r**2))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + ((n1-n2)*(k1*r**2))/(k1*r**2+2**n1)*sv_trajectory_plots[:,SV12_i]**2 \
                               + (2**n1-n1)*sv_trajectory_plots[:,SV21_i]**2 )**0.5
  else:
      predicted_sys_non_norm = np.zeros(predicted_sys_norm.shape)
      predicted_non_non_norm = np.zeros(predicted_sys_norm.shape)

  quad_norms = [predicted_sys_sys_norm,predicted_non_sys_norm,predicted_sys_non_norm,predicted_non_non_norm]
  
  return sv_trajectory_plots, predicted_sys_norm, predicted_non_norm, quad_norms, U_preds, V_T_preds, sv_indices

def dynamic(n1, n2, k1, k2, r, X, Y, U, VT, a_init, num_time_steps, num_svds, sv_indices):
  # Returns predictions of the training dynamics of the linear net using the final SVs predicted and stored from prev generations.

  # Retrieve the correctly ordered input-output covariance SVs (predicted last gen), and predict the input covariance SVs since they don't
  # change over generations
  _, final_svds_s_only, _ = svs((1/X.shape[1])*(Y @ X.T), U, VT, num_svds, k2)
  sxv1 = (k1*r**2+2**n1)/2**(n1)
  sxv3 = (k1*r**2)/(2**n1)
  sxv_preds_inv = sxv1*np.eye(n1)
  sxv_preds_inv = [1/sxv_preds_inv[i,i] for i in range(sxv_preds_inv.shape[0]) if i < len(final_svds_s_only) and final_svds_s_only[i]]
  if k2 > 0:
      for _ in range(2**n1 - n1): sxv_preds_inv.append(1/sxv3)
  d_inv = np.array(sxv_preds_inv)

  # Get the correct indices for each unique SV from the previoys generations (used to calculate norm trajectories using unqiue SVs trajects)
  SV11_i, SV12_i, SV21_i = sv_indices

  # Setting up time indices array and exponential components of SV dynamics formula
  taus = (np.arange(0,num_time_steps,1).reshape(num_time_steps,1)/tau)
  exp_s = np.exp(-2*final_svds_s_only[:num_svds]*taus)

  # Predict SV Trajectories using dynamics formula and predicted SVs from previous generation
  numerator = final_svds_s_only*d_inv
  denominator = 1 - (1 - (final_svds_s_only*d_inv/a_init))*exp_s
  sv_trajectory_plots = np.zeros(denominator.shape)
  num_sig_svs = numerator[numerator > 0].shape[0]
  sv_trajectory_plots[:,:num_sig_svs] = numerator[:num_sig_svs].reshape(1, num_sig_svs)/denominator[:,:num_sig_svs]

  # Predict Frobenius Norm Trajectories using SV dynamics (partitioned by output only, ie: full input used for mappings)
  predicted_sys_norm = ( (n2*2**n1)/(k2*r**2+2**n1)*sv_trajectory_plots[:,SV11_i]**2)**0.5
  if k2 > 0:
      predicted_non_norm = ( (k2*n2*r**2/(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + (n1-n2)*sv_trajectory_plots[:,SV12_i]**2 \
                               + (2**n1-n1)*sv_trajectory_plots[:,SV21_i]**2 )**0.5
  else:
      predicted_non_norm = np.zeros(predicted_sys_norm.shape)
      
  # Predict Frobenius Norm Trajectories using SV dynamics (partitioned by input and output)
  predicted_sys_sys_norm = ( (n2*2**(2*n1))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2)**0.5
  predicted_non_sys_norm = ( (n2*2**n1*(k1*r**2))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2)**0.5

  if k2 > 0:
      predicted_sys_non_norm = ( (k2*n2*r**2*2**n1)/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + ((n1-n2)*2**n1)/(k1*r**2+2**n1)*sv_trajectory_plots[:,SV12_i]**2 )**0.5
      predicted_non_non_norm = ( (k2*n2*r**2*(k1*r**2))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + ((n1-n2)*(k1*r**2))/(k1*r**2+2**n1)*sv_trajectory_plots[:,SV12_i]**2 \
                               + (2**n1-n1)*sv_trajectory_plots[:,SV21_i]**2 )**0.5
  else:
      predicted_sys_non_norm = np.zeros(predicted_sys_norm.shape)
      predicted_non_non_norm = np.zeros(predicted_sys_norm.shape) 

  quad_norms = [predicted_sys_sys_norm,predicted_non_sys_norm,predicted_sys_non_norm,predicted_non_non_norm]
 
  return sv_trajectory_plots, predicted_sys_norm, predicted_non_norm, quad_norms

def init_random_params(scale, layer_sizes, seed):
  # Returns a list of tuples where each tuple is the weight matrix and bias vector for a layer
  np.random.seed(seed)
  return [np.random.normal(0.0, scale, (n, m)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

@jit
def predict(params, inputs):
  # Propagate data forward through the network
  return jnp.dot(params[1], jnp.dot(params[0], inputs))

@jit
def loss(params, batch):
  # Loss over a batch of data
  inputs, targets = batch
  preds = predict(params, inputs)
  return np.mean(np.sum(jnp.power((preds - targets),2), axis=1))

if __name__ == "__main__":

  def updater(params, batch):
    # Function which updates model parameters
    sigma_xx = (1/batch[0].shape[1])*np.dot(batch[0], batch[0].T)
    sigma_yx = (1/batch[0].shape[1])*np.dot(batch[1], batch[0].T)
    W2W1 = np.dot(params[1], params[0])
    return [params[0] + ((1/tau) * np.dot(params[1].T, sigma_yx - np.dot(W2W1, sigma_xx))),\
            params[1] + ((1/tau) * np.dot(sigma_yx - np.dot(W2W1, sigma_xx), params[0].T))]

  # Data Hyper-parameters
  n1 = 3 #n1 num sys inputs
  n2 = 1 #n2 num sys outputs
  k1 = 3 #k1 num nonsys reps input
  k2 = 1  #k2 num nonsys reps output
  r = 2 #r scale

  # Training hyper-parameters
  num_hidden = 50.0
  layer_sizes = [n1 + k1*2**n1,int(num_hidden), n2 + k2*2**n1]
  step_size = 0.02
  num_epochs = 35
  param_scale = 0.01/float(num_hidden)
  num_generations = 15
  if k2 > 0:
      num_svds = 2**n1
  else:
      num_svds = n2
  seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init

  # Holds the SV trajectories to be plotted
  trainings = np.zeros((num_epochs, num_svds, num_generations))
  predictions = np.zeros((num_epochs, num_svds, num_generations))
  sys_norms = np.zeros((num_epochs, num_generations))
  non_norms = np.zeros((num_epochs, num_generations))
  sys_sys_norms = np.zeros((num_epochs, num_generations))
  sys_non_norms = np.zeros((num_epochs, num_generations))
  non_sys_norms = np.zeros((num_epochs, num_generations))
  non_non_norms = np.zeros((num_epochs, num_generations))
  preds_sys_norms = np.zeros((num_epochs, num_generations))
  preds_non_norms = np.zeros((num_epochs, num_generations))
  preds_sys_sys_norms = np.zeros((num_epochs, num_generations))
  preds_sys_non_norms = np.zeros((num_epochs, num_generations))
  preds_non_sys_norms = np.zeros((num_epochs, num_generations))
  preds_non_non_norms = np.zeros((num_epochs, num_generations))
  losses = np.zeros((num_epochs, num_generations))

  # Create Dataset training data
  X = np.flip(gen_binary_patterns(n1).T, axis=1)
  for i in range(k1):
      X = np.vstack([X, r*np.eye(2**n1)])

  # Create Dataset labels
  Y = np.flip(gen_binary_patterns(n1).T, axis=1)
  for i in range(k2):
      Y = np.vstack([Y, r*np.eye(2**n1)])
  Y_keep_feat = np.arange(Y.shape[0])
  Y_delete = np.random.choice(n1, n1-n2, replace=False)
  Y_keep_feat = np.delete(Y_keep_feat, Y_delete)
  Y = Y[Y_keep_feat]
  print("Input Data: \n", X)
  print("Initial Labels: \n", Y)
  batch_size = X.shape[1]
  start_labels = np.copy(Y)

  # Variable defined the same as in the paper, used in dynamics formula
  tau = 1/(X.shape[1]*step_size)

  for i in range(num_generations):
      # Initialize network parameters and initial network SVs set to be small
      params = init_random_params(param_scale, layer_sizes, seed)
      a = np.ones(num_svds)
      a[:n1] = a[:n1]*7e-7
      a[n1:] = a[n1:]*8e-7
      traj_real = a
      
      # Get ground truth trajectories and save them for plotting at end, as well as useful matrices and SV indices for next generation
      if i == 0:
          predictions[:,:,i], preds_sys_norms[:,i], preds_non_norms[:,i], preds_quad_norms, U, VT, sv_indices =\
                                     dynamic_formula(n1, n2, k1, k2, r, a.reshape(1,num_svds), num_epochs, num_svds)
          preds_sys_sys_norms[:,i] = preds_quad_norms[0]
          preds_non_sys_norms[:,i] = preds_quad_norms[1]
          preds_sys_non_norms[:,i] = preds_quad_norms[2]
          preds_non_non_norms[:,i] = preds_quad_norms[3]
      else:
          predictions[:,:,i], preds_sys_norms[:,i], preds_non_norms[:,i], preds_quad_norms =\
                                      dynamic(n1, n2, k1, k2, r, X, Y, U, VT, a.reshape(1,num_svds), num_epochs, num_svds, sv_indices)
          preds_sys_sys_norms[:,i] = preds_quad_norms[0]
          preds_non_sys_norms[:,i] = preds_quad_norms[1]
          preds_sys_non_norms[:,i] = preds_quad_norms[2]
          preds_non_non_norms[:,i] = preds_quad_norms[3]

      # Run generations of IL. We track the SVs of the input-output covariance matrix from the network at the end of each epoch
      print("Generation: ", i)
      for epoch in range(num_epochs):
          for batch_start in range(0,X.shape[1],batch_size):
              params = updater(params, (X[:,batch_start:batch_start+batch_size],Y[:,batch_start:batch_start+batch_size]))
          lossr = loss(params, (X,Y))
          losses[epoch, i] = lossr
          print(lossr)
 
          # Get input-output mapping of training network and its singular values
          full_map = np.dot(params[1], params[0])
          _, a, _ = svs(full_map, U, VT, num_svds, k2)

          traj_real = np.vstack([traj_real, a])
          
          # Get systematic and non-systematic frobenius norms
          sys_norms[epoch,i] = np.linalg.norm(full_map[:n2],'fro')
          non_norms[epoch,i] = np.linalg.norm(full_map[n2:],'fro')

          # Get Quadrant frobenius norms
          sys_sys_norms[epoch,i] = np.linalg.norm(full_map[:n2,:n1],'fro')
          sys_non_norms[epoch,i] = np.linalg.norm(full_map[n2:,:n1],'fro')
          non_sys_norms[epoch,i] = np.linalg.norm(full_map[:n2,n1:],'fro')
          non_non_norms[epoch,i] = np.linalg.norm(full_map[n2:,n1:],'fro')
      
      # Get network generated labels and network SVs at end of current generation
      Y = predict(params,X)
      full_map = np.dot(params[1], params[0])
      _, s, _ = svs(full_map, U, VT, num_svds, k2)
      print("Generation "+str(i)+" SVs: ", s)
      trainings[:,:,i] = traj_real[1:]

  # Plot each trajectory. 
  # Rows are time-steps, Columns are full trajectories
  print("Num Plots: ",trainings.shape[2])
  for j in range(trainings.shape[1]):
    for k in range(trainings.shape[2]):
        if k == 0:
            plt.plot(np.arange(num_epochs), predictions[:,j,k], color='red', label='Prediction', linestyle='dashed')
            plt.plot(np.arange(num_epochs), trainings[:,j,k], color='blue', label='Real')
        else:
            plt.plot(np.arange(num_epochs), predictions[:,j,k], color='red', linestyle='dashed')
            plt.plot(np.arange(num_epochs), trainings[:,j,k], color='blue')
        plt.legend()
    #plt.title("Deep Dynamics of Non-systematic Singular Value " + str(j))
    plt.ylabel("Singular Value")
    plt.xlabel("Epoch number")
    plt.grid()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    #plt.savefig(str(j)+".png")
    plt.close()
  
  # Plot each unqiue trajectory for comparison.
  print(trainings.shape)
  colours_arr = [('red','green',r'Prediction $\pi_1$', r'Real $\pi_1$'),('blue','orange',r'Prediction $\pi_2$', r'Real $\pi_2$'),
                 ('cyan','purple',r'Prediction $\pi_3$', r'Real $\pi_3$')]
  unique_svs,unique_indices = np.unique(predictions[-1,:,0], return_index=True)
  unique_indices = unique_indices[::-1]
  unique_svs = np.round(unique_svs,2)[::-1]
  colours = {unique_svs[i]:colours_arr[i] for i in range(unique_svs.shape[0])}
  print("Num Plots: ", trainings.shape[2])
  for j in [unique_indices[0],unique_indices[2],unique_indices[1]]:
      identifier = colours[np.round(predictions[-1,j,0],2)]
      print(identifier)
      labelr = identifier[2]
      labelr_real = identifier[3]
      plt.plot(np.arange(num_epochs), predictions[:,j,0], color=identifier[0], label=labelr, linestyle='dashed')
      plt.plot(np.arange(num_epochs), trainings[:,j,0], color=identifier[1], label=labelr_real)
  #plt.title("Deep Dynamics of Singular Values")
  plt.ylabel("Singular Value")
  plt.xlabel("Epoch number")
  plt.grid()
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.legend(loc='center', bbox_to_anchor=(0.5, 0.7))
  #plt.savefig("SVs.png")
  plt.close()

  # Plot losses
  plt.plot(losses.T.reshape(-1))
  #plt.title("Network Loss Over Training Multiple Generations")
  plt.ylabel("Loss")
  plt.xlabel("Epoch number")
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.grid()
  #plt.savefig("losses.png")
  plt.close()

  # Plot frobenius norms
  for k in range(sys_norms.shape[1]):
      if k == 0:
          plt.plot(sys_norms[:,k], color='red', label=r'$\Omega_y$-Norm')
          plt.plot(preds_sys_norms[:,k], color='green', label=r'Predicted $\Omega_y$-Norm', linestyle='dashed')
          plt.plot(non_norms[:,k], color='blue', label=r'$\Gamma_y$-Norm')
          plt.plot(preds_non_norms[:,k], color='orange', label=r'Predicted $\Gamma_y$-Norm', linestyle='dashed')
      else:
          plt.plot(sys_norms[:,k], color='red')
          plt.plot(preds_sys_norms[:,k], color='green', linestyle='dashed')
          plt.plot(non_norms[:,k], color='blue')
          plt.plot(preds_non_norms[:,k], color='orange', linestyle='dashed')
  #plt.title("Deep Dynamics of Frobenius Norms")
  plt.ylim([-0.05, np.max([sys_norms,non_norms])+0.25])
  plt.xlim([-10, num_epochs])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Frobenius Norm")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper left', bbox_to_anchor=(0.0, 0.52, 0.5, 0.5))
  plt.grid()
  #plt.savefig('norms_plot.png')
  plt.close()
  np.savetxt('dense_norm_v_gen.txt', non_norms[-1,:])

  # Plot Systematic Quadrant norms
  for k in range(sys_norms.shape[1]):
      if k == 0:
          plt.plot(sys_sys_norms[:,k], color='red', label=r'$\Omega_x\Omega_y$-Norm')
          plt.plot(preds_sys_sys_norms[:,k], color='green', label=r'Predicted $\Omega_x\Omega_y$-Norm', linestyle='dashed')
          plt.plot(non_sys_norms[:,k], color='blue', label=r'$\Gamma_x\Omega_y$-Norm')
          plt.plot(preds_non_sys_norms[:,k], color='orange', label=r'Predicted $\Gamma_x\Omega_y$-Norm', linestyle='dashed')
      else:
          plt.plot(sys_sys_norms[:,k], color='red')
          plt.plot(preds_sys_sys_norms[:,k], color='green', linestyle='dashed')
          plt.plot(non_sys_norms[:,k], color='blue')
          plt.plot(preds_non_sys_norms[:,k], color='orange', linestyle='dashed')
  #plt.title("Deep Dynamics of Quadrants Frobenius Norms")
  plt.ylim([-0.05, np.max([sys_sys_norms,non_sys_norms])+0.45])
  plt.xlim([-10, num_epochs])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Frobenius Norm")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper left', bbox_to_anchor=(0.0, 0.52, 0.5, 0.5))
  plt.grid()
  #plt.savefig('sys_quad_norms_plot.png')
  plt.close()

  # Plot Non-systematic Quadrant norms
  for k in range(sys_norms.shape[1]):
      if k == 0:
          plt.plot(sys_non_norms[:,k], color='purple', label=r'$\Omega_x\Gamma_y$-Norm')
          plt.plot(preds_sys_non_norms[:,k], color='lime', label=r'Predicted $\Omega_x\Gamma_y$-Norm', linestyle='dashed')
          plt.plot(non_non_norms[:,k], color='navy', label=r'$\Gamma_x\Gamma_y$-Norm')
          plt.plot(preds_non_non_norms[:,k], color='cyan', label=r'Predicted $\Gamma_x\Gamma_y$-Norm', linestyle='dashed')
      else:
          plt.plot(sys_non_norms[:,k], color='purple')
          plt.plot(preds_sys_non_norms[:,k], color='lime', linestyle='dashed')
          plt.plot(non_non_norms[:,k], color='navy')
          plt.plot(preds_non_non_norms[:,k], color='cyan', linestyle='dashed')
  #plt.title("Deep Dynamics of Quadrants Frobenius Norms")
  plt.ylim([-0.05, np.max([sys_non_norms,non_non_norms])+0.25])
  plt.xlim([-10, num_epochs])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Frobenius Norm")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper left', bbox_to_anchor=(0.0, 0.52, 0.5, 0.5))
  plt.grid()
  #plt.savefig('non_quad_norms_plot.png')
  plt.close()

  # Plot Quadrant norms together
  for k in range(sys_norms.shape[1]):
      if k == 0:
          plt.plot(sys_sys_norms[:,k], color='red', label=r'$\Omega_x\Omega_y$-Norm')
          plt.plot(preds_sys_sys_norms[:,k], color='green', label=r'Predicted $\Omega_x\Omega_y$-Norm', linestyle='dashed')
          plt.plot(non_sys_norms[:,k], color='blue', label=r'$\Gamma_x\Omega_y$-Norm')
          plt.plot(preds_non_sys_norms[:,k], color='orange', label=r'Predicted $\Gamma_x\Omega_y$-Norm', linestyle='dashed')
          plt.plot(sys_non_norms[:,k], color='purple', label=r'$\Omega_x\Gamma_y$-Norm')
          plt.plot(preds_sys_non_norms[:,k], color='lime', label=r'Predicted $\Omega_x\Gamma_y$-Norm', linestyle='dashed')
          plt.plot(non_non_norms[:,k], color='navy', label=r'$\Gamma_x\Gamma_y$-Norm')
          plt.plot(preds_non_non_norms[:,k], color='cyan', label=r'Predicted $\Gamma_x\Gamma_y$-Norm', linestyle='dashed')
      else:
          plt.plot(sys_sys_norms[:,k], color='red')
          plt.plot(preds_sys_sys_norms[:,k], color='green', linestyle='dashed')
          plt.plot(non_sys_norms[:,k], color='blue')
          plt.plot(preds_non_sys_norms[:,k], color='orange', linestyle='dashed')
          plt.plot(sys_non_norms[:,k], color='purple')
          plt.plot(preds_sys_non_norms[:,k], color='lime', linestyle='dashed')
          plt.plot(non_non_norms[:,k], color='navy')
          plt.plot(preds_non_non_norms[:,k], color='cyan', linestyle='dashed')
  #plt.title("Deep Dynamics of Quadrants Frobenius Norms")
  plt.ylim([-0.05, np.max([sys_sys_norms,non_sys_norms,sys_non_norms,non_non_norms])+0.45])
  plt.xlim([-10, num_epochs])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Frobenius Norm")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper left', bbox_to_anchor=(0.0, 0.52, 0.5, 0.5))
  plt.grid()
  #plt.savefig('quad_norms_plot.png')
  plt.close()

  print("Start Labels")
  print(start_labels)
  print("End Labels")
  print(Y)
