# reinforce_cp: REINFORCE algorithm with CartPole-v0 
# Christian Hubbs
# 10.03.2018

# Use this to test efficacy of REINFORCE and other tools

import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import sys
# sys.path.insert(0, r'C:\Users\u757585\Documents\AJK_Personal_Files\2018\DTA_ReinforcementLearning\dow-alpha\Deprecated')
from reinforce_policy_gradients import *
from rl_utils import *

start_time = time.time()
# Generate settings to search network architectures
search_dict = generate_search_dict(1, 4, 2, value_estimator=True, convergence_tol=1e-3)

env = gym.make("CartPole-v0")

# Cycle through architectures
for arch in search_dict:
	tf.reset_default_graph()
	sess = tf.Session()

	# Build networks
	policy_estimator = policyEstimator(sess, env, 
		n_hidden_layers=search_dict[arch]['n_hidden_layers'],
		n_hidden_nodes=search_dict[arch]['n_hidden_nodes'], 
		learning_rate=search_dict[arch]['learning_rate'])

	if search_dict[arch]['value_estimator']:
		value_estimator = valueEstimator(policy_estimator, 
			n_hidden_layers=search_dict[arch]['n_hidden_layers'],
			n_hidden_nodes=search_dict[arch]['n_hidden_nodes'], 
			learning_rate=search_dict[arch]['learning_rate'])

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Run REINFORCE
	if search_dict[arch]['value_estimator']:
		eps_pol, smoothed_pol = reinforce(env, policy_estimator, 
			value_estimator, settings=search_dict[arch])
	else:
		eps_pol, smoothed_pol = reinforce(env, policy_estimator,
			settings=search_dict[arch])

	print("Training Complete:")
	print(np.mean(smoothed_pol[-10:]))
	plt.figure(figsize=(12,8))
	plt.plot(smoothed_pol)
	plt.savefig(str(search_dict[arch]['data_path'] + '/plot.png'))

end_time = time.time()
print("Total elapsed time:")
print(end_time - start_time)







