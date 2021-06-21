import os

import numpy as np
import matplotlib.pyplot as plt


def rmse(gt, pred):

	"""Calculte the Root Mean Squared Error between two tensors representing collection of
	points in 3D space"""

	result = np.mean(np.sqrt(np.sum((gt - pred) ** 2, axis=-1)))

	return result


def count_components(data, radius):

	"""Find the numner of connected components in the spatial graph defined by the 3D points in data,
	where the existence of an edge between two nodes (points) is determined by their proximity and
	a radius parameter"""

	components = {}
	counter = 0

	components[counter] = {'elements': [], 'center': None}
	candidate = data[0]
	components[counter]['elements'].append(candidate)
	data = np.delete(data, 0, axis=0)

	while len(data) != 0:

		element_added = False

		for candidate_idx, candidate in enumerate(data):
			anchors = np.array(components[counter]['elements'])
			candidate_tiled = np.tile(candidate, (len(anchors), 1))


			distances = np.linalg.norm(anchors - candidate_tiled, axis=-1)

			# the candidate node is close to at least one element already in the component
			# add it to the component, delete it from data and loop through the remaining candidates
			if np.any(distances <= radius):
				components[counter]['elements'].append(candidate)
				data = np.delete(data, candidate_idx, axis=0)
				element_added = True
				break

		# if we didn't add an element, initialise a new component
		# otherwise keep looking through the next candidates from data
		if not element_added:
			counter += 1

			if len(data) != 0:
				components[counter] = {'elements': [], 'center': None}
				candidate = data[0]
				components[counter]['elements'].append(candidate)
				data = np.delete(data, 0, axis=0)

	for key, component in components.items():
		components[key]['center'] = np.mean(components[key]['elements'], axis=0)

	return components


def IoU(gt, pred, resolution):

	"""Intersection-over-Union (IoU, Jaccard Index) algorithm
	source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2 """

	start = -1
	end = 1
	n = int((end - start) / resolution)
	grid = np.linspace(start, end, n)

	gt_mask = np.zeros((n,n,n))
	pred_mask = np.zeros((n,n,n))

	for particle in gt:
		x, y, z = particle

		x_idx = np.argwhere(grid < x)[-1, -1]
		y_idx = np.argwhere(grid < y)[-1, -1]
		z_idx = np.argwhere(grid < z)[-1, -1]

		gt_mask[x_idx, y_idx, z_idx] = 1

	for particle in pred:
		x, y, z = particle

		x_idx = np.argwhere(grid < x)[-1, -1]
		y_idx = np.argwhere(grid < y)[-1, -1]
		z_idx = np.argwhere(grid < z)[-1, -1]

		pred_mask[x_idx, y_idx, z_idx] = 1


	intersection_mask = gt_mask * pred_mask
	union_mask = gt_mask + pred_mask
	union_mask[union_mask == 2] = 1

	return np.sum(intersection_mask) / np.sum(union_mask)


def plot_velocities_profile_metric_rollout(vel_next_arr, title=None, show=False, save_data=None):

    """Ground-truth velocities profile: Histogram of magnitude of velcoties during rollout"""

    if save_data is None:
        save_data = []

    vel_next_arr = np.array(vel_next_arr)
    print("vel_next_arr shape = ", vel_next_arr.shape)
    # source: https://stackoverflow.com/questions/9171158/how-do-you-get-the-magnitude-of-a-vector-in-numpy
    vel_next_magn = [ np.linalg.norm(x) for v_n_arr in vel_next_arr for x in v_n_arr ]
    print("vel_next_magn len = ", len(vel_next_magn))

    fig, ax = plt.subplots(1, 1)

    ax.hist(vel_next_magn)
    ax.set_title(title)
    ax.set_xlabel('Velocity magnitude')
    ax.set_ylabel('Occurences in rollout')

    if len(save_data):
        # save_data format: save_data=[des_dir, name, label, args.env + args.special_data, args.eval_data_f])
        assert(len(save_data) == 5)
        plt.savefig(os.path.join(save_data[0], '%s_%d_%s_%s' % (save_data[1], save_data[2], save_data[3], save_data[4])))

    if show:
        plt.show()
