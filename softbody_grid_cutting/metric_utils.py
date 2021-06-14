import numpy as np

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