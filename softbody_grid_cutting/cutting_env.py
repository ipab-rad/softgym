import os
import os.path as osp
import argparse
import numpy as np
import cv2
import pyflex
from tqdm import tqdm
import h5py
import json
from scipy.spatial.transform import Rotation
import random as rnd

from softgym.utils.pyflex_utils import center_object

from metric_utils import rmse, count_components, IoU, plot_velocities_profile_metric_rollout


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


class CutEnv(object):

    """ The class describing a cutting environment experiment and organizing its functionality """

    def __init__(self, args):

        """ Initialize environment variables """

        self.dim_shape_state = 14 # for each gripper finger keep curr and prev states => [x,y,z,x_prev,y_prev,z_prev,qx,qy,qz,qw,qx_prev,qy_prev,qz_prev,qw_prev]

        self.headless = int(not(args.render))
        self.render = args.render
        self.save_frames = args.save_frames

        self.render_mode = args.render_mode
        self.window_size = args.img_size

        pyflex.init(self.headless, self.render, self.window_size, self.window_size)

        self.dimx = args.dimx
        self.dimy = args.dimy
        self.dimz = args.dimz

        self.p_radius = args.p_radius
        self.pulling_dist = args.pulling_dist

        self.spring_cut_step = args.spring_cut_step
        self.remove_duplicate_springs = args.remove_duplicate_springs

        self.sample_1D_obj = args.sample_1D_obj
        self.sample_2D_obj = args.sample_2D_obj
        self.sample_3D_obj = args.sample_3D_obj
        self.sample_knife = args.sample_knife

        self.p_radius = args.p_radius

        if args.type_of_cut.lower() == "full" or args.type_of_cut.lower() == "full cut":
            self.type_of_cut = "full"
        elif args.type_of_cut.lower() == "partial" or args.type_of_cut.lower() == "partial cut":
            self.type_of_cut = "partial"
        elif args.type_of_cut.lower() == "no" or args.type_of_cut.lower() == "no cut" or args.type_of_cut.lower() == "none":
            self.type_of_cut = "none"
        elif args.type_of_cut.lower() == "just cut":
            self.type_of_cut = "just cut"

        self.sharp_knife = args.sharp_knife
        self.cut_ratio = args.cut_ratio

        self.stiff_body = args.stiff_body

        self.rejection_sampling_ratio = args.rejection_sampling_ratio


    def check_intersect(self, line, plane):

        """ Check if a line (defined by two points), intersects a plane (defined by 3 points),
            source: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection#Parametric_form """

        la, lb = line
        p0, p1, p2 = plane

        lab = lb - la
        p01 = p1 - p0
        p02 = p2 - p0

        A = np.array([-lab, p01, p02])
        B = np.array(la - p0)

        try:
            A_inv = np.linalg.inv(A.T)
        except:
            return False, False

        t, u, v = np.matmul(A_inv, B.T).T

        return True, (t >= 0 and t <= 1 and u >= 0 and u <= 1 and v >= 0 and v <= 1 and (u+v) <= 1)


    def produce_cutting_mask(self, spring_indices, knife_half_edge, spring_list, spring_dict):

        """ Produce a cutting mask, given a list of springs and a plane describing a knife blade.
            (Basically, which spring-points lines intersect the knife blade's plane) """

        shape_states_ = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)

        knife_center = shape_states_[0][:3]
        knife_quat = shape_states_[0][6:10]
        rot = Rotation.from_quat(knife_quat)

        knife_plane_1 = np.array([knife_center + np.array([0, knife_half_edge[1], knife_half_edge[2]]),
                                knife_center + np.array([0, -knife_half_edge[1], knife_half_edge[2]]),
                                knife_center + np.array([0, -knife_half_edge[1], -knife_half_edge[2]])])

        # account for varying knife orientation when considering the knife plane
        # translate to origin - rotate - translate back to original position
        knife_plane_1 -= knife_center
        knife_plane_1 = rot.apply(knife_plane_1)
        knife_plane_1 += knife_center

        knife_plane_2 = np.array([knife_center + np.array([0, knife_half_edge[1], -knife_half_edge[2]]),
                                knife_center + np.array([0, -knife_half_edge[1], knife_half_edge[2]]),
                                knife_center + np.array([0, -knife_half_edge[1], -knife_half_edge[2]])])

        knife_plane_2 -= knife_center
        knife_plane_2 = rot.apply(knife_plane_2)
        knife_plane_2 += knife_center

        springs_n = spring_indices.shape[0]

        particle_positions = pyflex.get_positions().reshape(-1, 4)

        cutting_mask = np.zeros(springs_n, dtype=np.int32)

        for spring_idx, spring in enumerate(spring_indices):
            left, right = spring

            spring_line = np.array([particle_positions[left, :3],
                                    particle_positions[right, :3]])

            intersect_plane_1, intersect_knife_1 = self.check_intersect(spring_line, knife_plane_1)
            intersect_plane_2, intersect_knife_2 = self.check_intersect(spring_line, knife_plane_2)

            if intersect_knife_1 or intersect_knife_2:
            # if intersect_knife_1:
                cutting_mask[spring_idx] = 1

                if (left, right) not in spring_dict or (right, left) not in spring_dict:
                    spring_dict[(left, right)] = None
                    spring_dict[(right, left)] = None

                    spring_list.append((left, right))

        return cutting_mask, spring_list, spring_dict


    def reset(self, rollout_dir=""):

        """ Reset environment under existing parameterization """

        if self.sample_1D_obj:
            self.dimx = 1
            self.dimy = 1
            self.dimz = rnd.randint(6.0, 20.0)
        elif self.sample_2D_obj:
            self.dimx = rnd.randint(2.0, 16.0)
            self.dimy = 1
            self.dimz = rnd.randint(2.0, 16.0)
        elif self.sample_3D_obj:
            self.dimx = rnd.randint(2.0, 16.0)
            self.dimy = rnd.randint(2.0, 16.0)
            self.dimz = rnd.randint(2.0, 16.0)

        if self.stiff_body:
            grid_stiffness = [0.8, 1, 0.9]  # stiff body
        else:
            grid_stiffness = [0.01, 0.01, 0.01]  # loose body

        self.config = {
            'GridPos': [0, 0, 0],
            'GridSize': [self.dimx, self.dimy, self.dimz],
            'GridStiff': grid_stiffness,
            'ParticleRadius': self.p_radius,
            'RenderMode': self.render_mode,
            'GridMass': 0.5,
            'SharpKnife': self.sharp_knife}

        scene_params = np.array([*self.config['GridPos'], *self.config['GridSize'], self.config['ParticleRadius'],
                                 *self.config['GridStiff'], self.config['RenderMode'], self.config['GridMass'],
                                 self.config['SharpKnife']])

        pyflex.set_scene(3, scene_params, 0)

        camera_param = {'pos': np.array([-0.15, 1, 1]),
                        'angle': np.array([0 * np.pi, -0.25 * np.pi, 0]),
                        'width': self.window_size,
                        'height': self.window_size}

        pyflex.set_camera_params(np.array([*camera_param['pos'], *camera_param['angle'], camera_param['width'], camera_param['height']]))

        center_object()

        # define the knife (currently a plane mesh object)
        if self.sample_knife:
            x = rand_float(0.0001, 0.001)
            y = rand_float(0.1, 0.4)
            z = rand_float(0.1, 0.5)
            self.knife_half_edge = np.array([x, y, z])
        else:
            if self.type_of_cut == "full" or self.type_of_cut == "partial" or self.type_of_cut == "just cut":
                ### FULL CUT / PARTIAL CUT / JUST CUT
                self.knife_half_edge = np.array([0.0005, 0.2, 0.2])
            elif self.type_of_cut == "none":
                ### NO CUT
                self.knife_half_edge = np.array([0.0005, 0.15, 0.1])

        # put the knife above the middle of the object-to-be-cut
        p_pos = pyflex.get_positions().reshape(-1, 4)
        p_n = pyflex.get_n_particles()

        self.init_knife_offset_Y = (self.knife_half_edge[1] + self.dimy * self.p_radius)

        self.knife_center = np.zeros(3)
        if self.type_of_cut == "full" or self.type_of_cut == "partial" or self.type_of_cut == "just cut":
            ### FULL CUT / PARTIAL CUT / JUST CUT
            self.knife_center[0] = np.random.uniform(low=p_pos[0, 0], high=p_pos[-1, 0], size=1)
            self.knife_center[1] += self.init_knife_offset_Y
            self.knife_center[2] += np.random.uniform(low=p_pos[0, 2], high=p_pos[-1, 2], size=1)
        elif self.type_of_cut == "none":
            ### NO CUT
            # source: https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
            R = np.sqrt(0.05)
            theta = np.random.random() * 2 * np.pi
            self.knife_center[0] += R * np.cos(theta)
            self.knife_center[1] = 0
            self.knife_center[2] += R * np.sin(theta)
        ###
        print("knife_center = ", self.knife_center)

        knife_orientation = np.zeros(3)
        if self.type_of_cut == "full" or self.type_of_cut == "partial" or self.type_of_cut == "just cut":
            ### FULL CUT / PARTIAL CUT / JUST CUT
            knife_orientation[1] = np.random.rand() * np.pi
        elif self.type_of_cut == "none":
            ### NO CUT
            knife_orientation[1] = np.random.rand() * 2 * np.pi
        ###
        print("knife_orientation = ", knife_orientation)

        # quat = np.array([0., 0., 0., 1.])
        rot = Rotation.from_euler('xyz', knife_orientation)
        self.quat = rot.as_quat()

        pyflex.add_box(self.knife_half_edge, self.knife_center, self.quat)

        self.pulling_vector = rot.apply(np.array([-1, 0, 0]))   # initially just [-1, 0, 0]
        print("pulling_vector = ", self.pulling_vector)
        # self.sawing_vector = rot.apply(np.array([0, 0, 1]))   # initially just [-1, 0, 0]

        self.config['TypeOfCut'] = self.type_of_cut
        self.config['CuttingRatio'] = self.cut_ratio
        self.config['StiffBody'] = self.stiff_body
        self.config['KnifeHalfEdge'] = self.knife_half_edge.tolist()
        self.config['KnifeCenter'] = self.knife_center.tolist()
        self.config['KnifeQuat'] = self.quat.tolist()
        self.config['KnifePullingVector'] = self.pulling_vector.tolist()

        with open(osp.join(rollout_dir, 'scene_params.json'), "w") as write_file:
            json.dump(self.config, write_file, indent=4)


    def generate_rollout(self, T=300, rollout_dir=""):

        """ Generate a simulation rollout """

        n_particles = pyflex.get_n_particles()
        n_shapes = pyflex.get_n_shapes()

        positions = np.zeros((T, n_particles + n_shapes, 3), dtype=np.float32)
        velocities = np.zeros((T, n_particles + n_shapes, 3), dtype=np.float32)
        shape_quats = np.zeros((T, n_shapes, 4), dtype=np.float32)

        offset_y_rand = np.random.rand()

        cut_spring_pairs_list = []
        cut_spring_pairs_dict = {}

        for t in tqdm(range(T)):

            shape_states_ = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)

            # # CUT - SAW -PULL
            # action_time_split = {'cut' : 0.15, 'saw' : 0.5, 'pull' : 1.0} # all phases sum to 1.0
            # cutting_step = self.init_knife_offset_Y / (T * action_time_split['cut'])
            # pulling_step = (self.pulling_dist * self.p_radius) / (T * action_time_split['pull'])
            # sawing_step = (1.5 * self.pulling_dist * self.p_radius) / (T * action_time_split['saw'])
            # knife_idx = 0 # the knife should be the only shape in the scene

            # if t > 0 and t < T * action_time_split['cut']:
            #     shape_states_ = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)
            #     shape_states_[knife_idx][3:6] = shape_states_[knife_idx][:3]
            #     shape_states_[knife_idx][1] -= cutting_step

            # if t > T * action_time_split['cut'] and t < T * action_time_split['saw']:
            #     shape_states_ = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)
            #     shape_states_[knife_idx][3:6] = shape_states_[knife_idx][:3]
            #     shape_states_[knife_idx][:3] -= sawing_step * self.sawing_vector

            # if t > T * action_time_split['saw'] and t < T:
            #     shape_states_ = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)
            #     shape_states_[knife_idx][3:6] = shape_states_[knife_idx][:3]
            #     shape_states_[knife_idx][:3] -= pulling_step * self.pulling_vector


            # CUT - PULL (FULL CUT / PARTIAL CUT)
            if self.type_of_cut == "full" or self.type_of_cut == "partial":
                # cutting speed
                cut_ratio = min([self.cut_ratio, 1.0])
                action_time_split = {'cut' : cut_ratio, 'pull' : 1.0} # all phases sum to 1.0
            # NO CUT & JUST PULL
            elif self.type_of_cut == "none":
                action_time_split = {'cut' : 0.0, 'pull' : 1.0} # all phases sum to 1.0
            # JUST CUT
            elif self.type_of_cut == "just cut":
                # cutting speed
                cut_ratio = self.cut_ratio
                action_time_split = {'cut' : cut_ratio, 'pull' : cut_ratio} # workaround so that we never pull a body part aside

            cutting_step = self.init_knife_offset_Y / (T * action_time_split['cut'])

            # tmp = (self.dimy + 5) * self.p_radius * offset_y_rand
            # cutting_step = tmp / (T * action_time_split['cut'])

            pulling_step = (self.pulling_dist * self.p_radius) / (T * action_time_split['pull'])
            knife_idx = 0 # the knife should be the only shape in the scene

            if t > 0 and t < T * action_time_split['cut']:
                shape_states_ = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)
                shape_states_[knife_idx][3:6] = shape_states_[knife_idx][:3]
                shape_states_[knife_idx][1] -= cutting_step

            if t > T * action_time_split['cut'] and t < T and self.type_of_cut != "just cut":
                shape_states_ = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)
                shape_states_[knife_idx][3:6] = shape_states_[knife_idx][:3]
                shape_states_[knife_idx][:3] -= pulling_step * self.pulling_vector

            pyflex.set_shape_states(shape_states_)


            if t % self.spring_cut_step == 0:

                if t == 0 and self.remove_duplicate_springs:

                    spring_indices = pyflex.get_spring_indices().reshape(-1, 2)
                    # print("len(spring_indices): ", len(spring_indices))

                    memo = {}
                    cut_mask = np.zeros(spring_indices.shape[0])
                    for idx_si, si in enumerate(spring_indices):
                        l = si[0]
                        r = si[1]

                        if l > r:
                            tmp = l
                            l = r
                            r = tmp

                        if (l,r) not in memo:
                            memo[(l, r)] = 1
                        else:
                            memo[(l, r)] += 1
                            cut_mask[idx_si] = 1

                    # tmp = list(memo.items())
                    # count = np.array([[x[0][0], x[0][1], x[1]] for x in tmp])
                    # print("# particle pairs with 1 spring: ", np.sum(count[:, 2] == 1))
                    # print("# particle pairs with 2 springs: ", np.sum(count[:, 2] == 2))
                    # print("# particle pairs with 3 springs: ", np.sum(count[:, 2] == 3))
                    # print("# particle pairs with 4 springs: ", np.sum(count[:, 2] == 4))

                    spring_indices_aug = np.concatenate((spring_indices, cut_mask[:, None]), axis=-1)
                    pyflex.cut_springs(spring_indices_aug)
                    pyflex.step()

                    # spring_indices = pyflex.get_spring_indices().reshape(-1, 2)
                    # print("len(spring_indices): ", len(spring_indices))

                    # memo = {}
                    # for si in spring_indices:
                    #     l = si[0]
                    #     r = si[1]

                    #     if l > r:
                    #         tmp = l
                    #         l = r
                    #         r = tmp

                    #     if (l,r) not in memo:
                    #         memo[(l, r)] = 1
                    #     else:
                    #         memo[(l, r)] += 1

                    # tmp = list(memo.items())
                    # count = np.array([[x[0][0], x[0][1], x[1]] for x in tmp])
                    # print("# particle pairs with 1 spring: ", np.sum(count[:, 2] == 1))
                    # print("# particle pairs with 2 springs: ", np.sum(count[:, 2] == 2))
                    # print("# particle pairs with 3 springs: ", np.sum(count[:, 2] == 3))
                    # print("# particle pairs with 4 springs: ", np.sum(count[:, 2] == 4))


                spring_indices = pyflex.get_spring_indices().reshape(-1, 2)
                cut_mask, cut_spring_pairs_list, cut_spring_pairs_dict = self.produce_cutting_mask(spring_indices,
                                                                                                   self.knife_half_edge,
                                                                                                   cut_spring_pairs_list,
                                                                                                   cut_spring_pairs_dict)
                spring_indices_aug = np.concatenate((spring_indices, cut_mask[:, None]), axis=-1)
                pyflex.cut_springs(spring_indices_aug)

            pyflex.step()

            # import time
            # time.sleep(0.1)

            if self.render:
                img = pyflex.render()

                if self.save_frames:

                    img = img.reshape(self.window_size, self.window_size, 4)[::-1, :, :3]  # Need to reverse the height dimension

                    img_path = os.path.join(rollout_dir, 'render_{0}.png'.format(str(t).zfill(4)))
                    cv2.imwrite(img_path, img[:, :, [2, 1, 0]])


            positions[t, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]
            shape_states = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)

            for k in range(n_shapes):
                positions[t, n_particles + k] = shape_states[k, :3]
                shape_quats[t, k] = shape_states[k, 6:10]

            if t > 0:
                dt = 1./60.
                velocities[t] = (positions[t] - positions[t - 1]) / dt

            scene_params = np.array([   *self.config['GridPos'], *self.config['GridSize'], self.config['ParticleRadius'], *self.config['GridStiff'],
                                        self.config['RenderMode'], self.config['GridMass'], *self.knife_half_edge, *self.knife_center, *self.quat]) # keep knife parameters as well, for scene reproduction
            data = [positions[t], velocities[t], shape_quats[t], scene_params, cut_spring_pairs_list]
            data_names = ['positions', 'velocities', 'shape_quats', 'scene_params', 'cut_spring_pairs']

            ### PARTIAL CUT
            if self.type_of_cut == "partial":
                if t == int(T * self.rejection_sampling_ratio):
                    comp = count_components(positions[t, :n_particles, :].copy(), self.p_radius * 1.25)
                    print("Mid-rollout # components: ", len(comp.keys()), comp.keys())
                    print([x['center'] for k,x in comp.items()])

                    if len(comp.keys()) > 1:    # full cut
                        print("\n\t\t\tREJECTED! FULL CUT!\n")
                        return np.zeros((T, n_particles + n_shapes, 3), dtype=np.float32), np.zeros((T, n_particles + n_shapes, 3), dtype=np.float32), np.zeros((T, n_shapes, 4), dtype=np.float32), 3
            elif self.type_of_cut == "none":
                ### NO CUT
                touch_body_flag = False
                if len(cut_spring_pairs_list):
                    touch_body_flag = True
                if t == 0 and touch_body_flag:
                    print("\n\t\t\tREJECTED! CUTS BODY!\n")
                    return np.zeros((T, n_particles + n_shapes, 3), dtype=np.float32), np.zeros((T, n_particles + n_shapes, 3), dtype=np.float32), np.zeros((T, n_shapes, 4), dtype=np.float32), 1
                if t == int(T * self.rejection_sampling_ratio) and np.array_equal(positions[t-2, :n_particles, :], positions[t, :n_particles, :]):
                    print("\n\t\t\tREJECTED! NOT PULLING BODY!\n")
                    return np.zeros((T, n_particles + n_shapes, 3), dtype=np.float32), np.zeros((T, n_particles + n_shapes, 3), dtype=np.float32), np.zeros((T, n_shapes, 4), dtype=np.float32), 2

            # print(cut_spring_pairs_list)

            store_data(data_names, data, os.path.join(rollout_dir, str(t) + '.h5'))

        return positions, velocities, shape_quats, 0


def main():

    """ The MAIN FUNCTION of our experiment """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--render', type=int, default=1, help='Whether to run the environment and render simulated scenes')
    parser.add_argument('--save_frames', type=int, default=0, help='Whether to save the rendered images to disk')
    parser.add_argument('--render_mode', type=int, default=1, help='Render mode: 1 - particle, 2 = cloth, 3 = both')

    parser.add_argument('--data_dir', type=str, default='./softbody_grid_cutting/data/', help='Path to the saved data')
    parser.add_argument('--img_size', type=int, default=300, help='Dimension for the rendered images')

    parser.add_argument('--dimx', type=int, default=16, help='Number of particles along the X axis')
    parser.add_argument('--dimy', type=int, default=1, help='Number of particles along the Y axis')
    parser.add_argument('--dimz', type=int, default=16, help='Number of particles along the Z axis')
    parser.add_argument('--p_radius', type=float, default=0.025, help='Interaction radius for the particles in the simulation')

    parser.add_argument('--n_rollout', type=int, default=5, help='Number of rollouts to be generated')
    parser.add_argument('--rollout_len', type=int, default=100, help='Length for each rollout')

    parser.add_argument('--spring_cut_step', type=int, default=2, help='Cut springs every Nth step of the simulation')
    parser.add_argument('--remove_duplicate_springs', type=int, default=1, help='Flag for whether to remove duplicate springs between the same pair of particles')

    parser.add_argument('--sample_1D_obj', type=int, default=0, help='Create 1D object by sampling its dimensions in the x axis for each rollout')
    parser.add_argument('--sample_2D_obj', type=int, default=0, help='Create 2D object by sampling its dimensions in the x and z axes for each rollout')
    parser.add_argument('--sample_3D_obj', type=int, default=0, help='Create 3D object by sampling its dimensions in the x, y and z axes for each rollout')
    parser.add_argument('--sample_knife', type=int, default=0, help='Create knife by sampling its half-edge dimensions in the x, y and z axes for each rollout')
    parser.add_argument('--pulling_dist', type=float, default=5, help='Number of particle radiuses the knife travels while pulling')

    parser.add_argument('--type_of_cut', type=str, default='full', help='Type of cut: (i) cut & pull: "full", "partial" or "none", \
                                                                        (ii) just cut: "just cut".')

    parser.add_argument('--sharp_knife', type=int, default=1, help='Sharp knife (if 1) or blunt knife (if 0)')

    parser.add_argument('--cut_ratio', type=float, default=0.3, help='Cutting timestep ratio [0.0, 1.0], definining the speed of the cut. \
                                                                    The rest (1.0 - cut_ratio) is the pulling ratio. E.g. 0.0: no cut & just pull, 1.0: just (a slow) cut & no pull.')

    parser.add_argument('--stiff_body', type=int, default=1, help='Simulate stiff body (if 1) or loose body (if 0)')

    parser.add_argument('--rejection_sampling_ratio', type=float, default=0.5, help='In which timestep of the simulation rollout would we do \
                                                                                    rejection sampling for partial cut or no cut. E.g. for 30 timestep rollouts, \
                                                                                    0.5 --> check at 15th timestep, 1.0 --> check at the end of the rollout.')

    args = parser.parse_args()

    env = CutEnv(args)

    stats = [init_stat(3), init_stat(3)]

    # for rollout_idx in range(args.n_rollout):
    rollout_idx = 0
    while rollout_idx < args.n_rollout:

        rollout_dir = os.path.join(args.data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)
        print("Geneating rollout # {0}, data folder: {1}".format(rollout_idx, rollout_dir))

        env.reset(rollout_dir=rollout_dir)

        positions, velocities, shape_quats, outcome_flag = env.generate_rollout(T=args.rollout_len, rollout_dir=rollout_dir)

        if outcome_flag == 0:   # all good
            rollout_idx += 1
        elif outcome_flag == 1: # Cutting body (touch in the first no cut rollout timestep)
            env.reset(rollout_dir=rollout_dir)
            continue
        elif outcome_flag == 2: # Not pulling body (halfway through a no cut & just pull rollout and body not touched)
            env.reset(rollout_dir=rollout_dir)
            print("Removing rollout # {0}, data folder: {1}".format(rollout_idx, rollout_dir))
            os.system('rm -rf ' + rollout_dir)
            continue
        elif outcome_flag == 3: # Full cut body (halfway through a partial cut rollout and body has been full cut)
            env.reset(rollout_dir=rollout_dir)
            print("Removing rollout # {0}, data folder: {1}".format(rollout_idx, rollout_dir))
            os.system('rm -rf ' + rollout_dir)
            continue

        datas = [positions.astype(np.float64), velocities.astype(np.float64)]

        # dist = rmse(positions[0], positions[-1])
        # print("RMSE: ", dist)

        # comp = count_components(positions[0, :-1, :].copy(), args.p_radius * 1.25)
        # print("Init # components: ", len(comp.keys()), comp.keys())
        # print([x['center'] for k,x in comp.items()])
        # print("\n")
        # comp = count_components(positions[-1, :-1, :].copy(), args.p_radius * 1.25)
        # print("Final # components: ", len(comp.keys()), comp.keys())
        # print([x['center'] for k,x in comp.items()])

        ### FULL CUT
        if args.type_of_cut.lower() == "full" or args.type_of_cut.lower() == "full cut":
            comp = count_components(positions[-1, :-1, :].copy(), args.p_radius * 1.25)
            print("Final # components: ", len(comp.keys()), comp.keys())
            print([x['center'] for k,x in comp.items()])

            if len(comp.keys()) < 2:    # partial cut
                print("\n\t\t\tREJECTED! PARTIAL CUT!\n")
                env.reset(rollout_dir=rollout_dir)
                print("Removing rollout # {0}, data folder: {1}".format(rollout_idx, rollout_dir))
                os.system('rm -rf ' + rollout_dir)
                rollout_idx -= 1    # undo addition
                continue

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0] * datas[j].shape[1]
            stats[j] = combine_stat(stats[j], stat)

    store_data(['positions', 'velocities'], stats, os.path.join(args.data_dir, 'stat.h5'))


if __name__ == '__main__':
    main()
