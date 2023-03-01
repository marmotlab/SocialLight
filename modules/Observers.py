from Utils import Utilities
import copy
import numpy as np


class VanillaObserver:
    def __init__(self, metaAgentID=0, num_agents=25, env=None, append_neighbour=False, args_dict=None):
        self.ID = metaAgentID
        self.num_agents = num_agents
        self.env = env
        self.append_neighbour = append_neighbour
        self.args_dict = args_dict
        self.share_horizon = self.args_dict['share_horizon']
        self.share_type = self.args_dict['share_type']

    def observe_single(self, index):
        raise NotImplementedError()

    def observe_all_single(self, indexes=None):
        observations = {}
        if indexes is None:
            for i in range(self.num_agents):
                observations[i] = self.observe_single(i)
        else:
            for index in indexes:
                observations[index] = self.observe_single(index)

        return observations

    def observe_all_complete(self, num_features=6, indexes=None, model_outputs=None):
        combined_observations = None
        single_observations = self.observe_all_single(indexes)
        if self.append_neighbour:
            combined_observations = self.share_neighbour_obs(single_observations, num_features=num_features)

        if self.share_horizon:
            if combined_observations is None:
                combined_observations = self.share_actions(single_observations, model_outputs)
            else:
                combined_observations = self.share_actions(combined_observations, model_outputs)
            return combined_observations

        if combined_observations is not None:
            return combined_observations
        return single_observations

    def share_actions(self, observations, model_outputs):
        for i in range(self.num_agents):
            observations[i] = np.reshape(observations[i], -1)
        if model_outputs is None:
            for i in range(self.num_agents):
                observations[i] = np.concatenate(
                    (observations[i], -1 * np.ones(4 * self.args_dict['horizon'] * self.args_dict['a_size'])))
            return observations
        else:
            for i in range(self.num_agents):
                neighbours = self.env.neighbour_dict['nt{}'.format(i + 1)]
                for neighbour in neighbours:
                    for j in range(self.args_dict['horizon']):
                        if neighbour is not None:
                            neigh_index = int(neighbour[2:]) - 1
                            neigh_prediction = np.reshape(model_outputs['forward_policies'][neigh_index][j], -1)
                            observations[i] = np.concatenate((observations[i], neigh_prediction))
                        else:
                            observations[i] = np.concatenate((observations[i], -1 * np.ones(self.args_dict['a_size'])))
            return observations

    def share_neighbour_obs(self, single_observations, num_features=3):
        combined_observations = copy.deepcopy(single_observations)
        for i in range(self.num_agents):
            neighbours = self.env.neighbour_dict['nt{}'.format(i + 1)]
            for neighbour in neighbours:
                if neighbour is not None:
                    neighbour_ind = int(neighbour[2:]) - 1
                    combined_observations[i] = np.concatenate(
                        (combined_observations[i], single_observations[neighbour_ind]))
                else:
                    combined_observations[i] = np.concatenate(
                        (combined_observations[i], -1 * np.ones((num_features * 6 + int(self.share_type) * 4))))
        return combined_observations


class TrafficObserver(VanillaObserver):
    def __init__(self, metaAgentID=0, num_agents=25, env=None, append_neighbour=False, args_dict=None):
        super().__init__(metaAgentID, num_agents, env, append_neighbour, args_dict)
        self.set_shape()

    def set_shape(self):
        if self.append_neighbour:
            if self.share_horizon:
                shape = [
                    180 + self.args_dict['a_size'] * self.args_dict['horizon'] * 4 + int(self.share_type) * 4 * 5]
            else:
                shape = [180 + int(self.share_type) * 4]
        elif self.share_horizon:
            shape = [36 + self.args_dict['a_size'] * self.args_dict['horizon'] * 4 + int(self.share_type) * 4]
        else:
            shape = [36 + int(self.share_type) * 4]
        self.shape, self.action_spaces, index_to_id, id_to_index = {}, {}, {}, {}
        for index, id in enumerate(self.env.agent_index):
            self.shape[id] = shape
            self.action_spaces[id] = self.args_dict['a_size']
            index_to_id[index], id_to_index[id] = id, index
        self.env.index_to_id, self.env.id_to_index = index_to_id, id_to_index

    def observe_single(self, index):
        ID = Utilities.index_to_ID(index, self.env)
        single_obs = np.reshape(self.env.TlsDict[ID].observe(), -1)
        if self.share_type:
            single_obs = np.concatenate((single_obs, self.env.agent_type_dict[index]))
        return single_obs

    def observe_all(self, indexes=None, model_outputs=None):
        return self.observe_all_complete(indexes=indexes, model_outputs=model_outputs, num_features=6)


class SmallObserver(VanillaObserver):
    def __init__(self, metaAgentID=0, num_agents=25, env=None, append_neighbour=False, args_dict=None):
        super().__init__(metaAgentID, num_agents, env, append_neighbour, args_dict)
        self.set_shape()

    def set_shape(self):
        if self.append_neighbour:
            if self.share_horizon:
                shape = [
                    90 + self.args_dict['a_size'] * self.args_dict['horizon'] * 4 + int(self.share_type) * 4 * 5]
            else:
                shape = [90 + +int(self.share_type) * 4 * 5]
        elif self.share_horizon:
            shape = [18 + self.args_dict['a_size'] * self.args_dict['horizon'] * 4 + int(self.share_type) * 4]
        else:
            shape = [18 + int(self.share_type) * 4]
        self.shape, self.action_spaces, index_to_id, id_to_index = {}, {}, {}, {}
        for index, id in enumerate(self.env.agent_index):
            self.shape[id] = shape
            self.action_spaces[id] = self.args_dict['a_size']
            index_to_id[index], id_to_index[id] = id, index
        self.env.index_to_id, self.env.id_to_index = index_to_id, id_to_index

    def observe_single(self, index):
        ID = Utilities.index_to_ID(index, self.env)
        single_obs = np.reshape(self.env.TlsDict[ID].small_observe(), -1)
        if self.share_type:
            single_obs = np.concatenate((single_obs, self.env.agent_type_dict[index]))
        return single_obs

    def observe_all(self, indexes=None, model_outputs=None):
        return self.observe_all_complete(indexes=indexes, model_outputs=model_outputs, num_features=3)


class MonacoObserver(VanillaObserver):
    def __init__(self, metaAgentID=0, num_agents=None, env=None, append_neighbour=False, args_dict=None):
        super(MonacoObserver, self).__init__(metaAgentID, num_agents, env, append_neighbour, args_dict)

        self.num_agents = len(self.env.agent_index)
        self.shape, self.action_spaces = {}, {}
        self.set_shape()

    def set_shape(self):
        """
        calculate the input dimension
        :return: self.shape (dict): key/tls_Id, value/list shape
        """
        index_to_id, id_to_index = {}, {}
        for index, id in enumerate(self.env.agent_index):
            if self.append_neighbour:
                shape = 3 * len(self.env.TlsDict[id].incoming_lane_list)
                for neighbor in self.env.TlsDict[id].neighbor_list:
                    shape += 3 * len(self.env.TlsDict[neighbor].incoming_lane_list)
                self.shape[id] = [shape]

            else:
                self.shape[id] = [3 * len(self.env.TlsDict[id].incoming_lane_list)]
            self.action_spaces[id] = len(self.env.TlsDict[id].action_space)
            index_to_id[index], id_to_index[id] = id, index
        self.env.index_to_id, self.env.id_to_index = index_to_id, id_to_index

    def observe_single(self, index):
        single_obs = np.reshape(self.env.TlsDict[index].observe(), -1)

        return single_obs

    def observe_all_single(self, indexes=None):
        observations = {}
        if indexes is None:
            for id in self.env.agent_index:
                observations[id] = self.observe_single(id)
        else:
            for id in indexes:
                observations[id] = self.observe_single(id)

        return observations

    def share_neighbour_obs(self, single_observations, num_features=3):
        combined_observations = copy.deepcopy(single_observations)
        for id in self.env.agent_index:
            neighbors = self.env.TlsDict[id].neighbor_list
            for neighbor in neighbors:
                combined_observations[id] = np.concatenate((combined_observations[id],
                                                            single_observations[neighbor]))
        return combined_observations

    def observe_all_complete(self, num_features=3, indexes=None, model_outputs=None):
        combined_observations = None
        single_observations = self.observe_all_single(indexes=self.env.agent_index)

        if self.append_neighbour:
            combined_observations = self.share_neighbour_obs(single_observations=single_observations,
                                                             num_features=3)

        if self.share_horizon:
            raise NotImplementedError

        if combined_observations is not None:
            return combined_observations

        else:
            return single_observations

    def observe_all(self, indexes=None, model_outputs=None):
        converted_obs = {}
        observations = self.observe_all_complete(indexes=indexes, model_outputs=model_outputs, num_features=3)
        for k, v in observations.items():
            converted_obs[self.env.id_to_index[k]] = v
        return converted_obs
class CityFlowObserver():
    def __init__(self, metaAgentID=0, env=None, append_neighbour=False, args_dict=None):
        super(CityFlowObserver, self).__init__()
        self.ID = metaAgentID
        self.env = env
        self.args_dict = args_dict

        self.append_neighbour = append_neighbour
        self.num_agents = len(self.env.agent_index)
        self.shape, self.action_spaces = {}, {}
        self.att = args_dict['ATTENTION'] if 'ATTENTION' in args_dict.keys() else False
        self.set_shape()

    def set_shape(self):
        for index, id in enumerate(self.env.agent_index):
            if not self.att:
                if self.append_neighbour:
                    self.shape[id] = [2 * 12 * 5]
                else:
                    self.shape[id] = [2 * 12]
            else:
                if self.append_neighbour:
                    self.shape[id] = [5, 24]
                else:
                    self.shape[id] = [2 * 12]

            self.action_spaces[id] = self.env.TlsDict[id].action_space_n

    def observe_single(self, index):
        ID = self.env.index_to_id[index]
        single_obs = np.reshape(self.env.TlsDict[ID].local_observe(), -1)
        return single_obs

    def observe_all_single(self, indexes=None):
        observations = {}
        if indexes is None:
            for i in range(self.num_agents):
                observations[i] = self.observe_single(i)
        else:
            for index in indexes:
                observations[index] = self.observe_single(index)

        return observations

    def share_neighbour_obs(self, single_observations, num_features):
        combined_observations = copy.deepcopy(single_observations)
        if self.att:
            for i in range(self.num_agents):
                id = self.env.index_to_id[i]
                neighbours = self.env.neighbour_dict[id]
                combined_observations[i] = [combined_observations[i]]
                for neighbour in neighbours:
                    if neighbour is not None:
                        neighbour_ind = self.env.id_to_index[neighbour]
                        combined_observations[i].append(single_observations[neighbour_ind])
                    else:
                        combined_observations[i].append(-1 * np.ones((num_features * 12)))

                combined_observations[i] = np.stack(combined_observations[i])
        else:
            for i in range(self.num_agents):
                id = self.env.index_to_id[i]
                neighbours = self.env.neighbour_dict[id]
                for neighbour in neighbours:
                    if neighbour is not None:
                        neighbour_ind = self.env.id_to_index[neighbour]
                        combined_observations[i] = np.concatenate(
                            (combined_observations[i], single_observations[neighbour_ind]))
                    else:
                        combined_observations[i] = np.concatenate(
                            (combined_observations[i], -1 * np.ones((num_features * 12))))

        return combined_observations

    def observe_all_complete(self, num_features=None, indexes=None):
        combined_observations = None
        single_observations = self.observe_all_single(indexes)
        if self.append_neighbour:
            combined_observations = self.share_neighbour_obs(single_observations=single_observations,
                                                             num_features=num_features)
        if combined_observations is not None:
            return combined_observations
        return single_observations

    def observe_all(self, indexes=None, model_outputs=None):
        return self.observe_all_complete(indexes=indexes, num_features=2)


class CityFlowV2Observer(CityFlowObserver):
    def __init__(self, metaAgentID=0, env=None, append_neighbour=False, args_dict=None):
        super(CityFlowV2Observer, self).__init__(
            metaAgentID=metaAgentID,
            env=env,
            append_neighbour=append_neighbour,
            args_dict=args_dict
        )
        self.ID = metaAgentID
        self.env = env
        self.args_dict = args_dict

        self.append_neighbour = append_neighbour
        self.num_agents = len(self.env.agent_index)
        self.shape, self.action_spaces = {}, {}
        self.att = args_dict['ATTENTION'] if 'ATTENTION' in args_dict.keys() else False
        self.set_shape()

    def set_shape(self):
        for index, id in enumerate(self.env.agent_index):
            if not self.att:
                if self.append_neighbour:
                    self.shape[id] = [20 * 5]
                else:
                    self.shape[id] = [20]
            else:
                if self.append_neighbour:
                    self.shape[id] = [5, 20]
                else:
                    self.shape[id] = [20]

            self.action_spaces[id] = self.env.action_space[id]

    def observe_single(self, index):
        ID = self.env.index_to_id[index]
        single_obs = np.reshape(self.env.local_observe(ID), -1)
        return single_obs

    def observe_all_single(self, indexes=None):
        observations = {}
        if indexes is None:
            for i in range(self.num_agents):
                observations[i] = self.observe_single(i)
        else:
            for index in indexes:
                observations[index] = self.observe_single(index)

        return observations

    def share_neighbour_obs(self, single_observations, num_features):
        combined_observations = copy.deepcopy(single_observations)
        if self.att:
            for i in range(self.num_agents):
                id = self.env.index_to_id[i]
                neighbours = self.env.neighbour_dict[id]
                combined_observations[i] = [combined_observations[i]]
                for neighbour in neighbours:
                    if neighbour is not None:
                        neighbour_ind = self.env.id_to_index[neighbour]
                        combined_observations[i].append(single_observations[neighbour_ind])
                    else:
                        combined_observations[i].append(-1 * np.ones((num_features * 20)))

                combined_observations[i] = np.stack(combined_observations[i])
        else:
            for i in range(self.num_agents):
                id = self.env.index_to_id[i]
                neighbours = self.env.neighbour_dict[id]
                for neighbour in neighbours:
                    if neighbour is not None:
                        neighbour_ind = self.env.id_to_index[neighbour]
                        combined_observations[i] = np.concatenate(
                            (combined_observations[i], single_observations[neighbour_ind]))
                    else:
                        combined_observations[i] = np.concatenate(
                            (combined_observations[i], -1 * np.ones((num_features * 20))))

        return combined_observations

    def observe_all_complete(self, num_features=None, indexes=None):
        combined_observations = None
        single_observations = self.observe_all_single(indexes)
        if self.append_neighbour:
            combined_observations = self.share_neighbour_obs(single_observations=single_observations,
                                                             num_features=num_features)
        if combined_observations is not None:
            return combined_observations
        return single_observations

    def observe_all(self, indexes=None, model_outputs=None):
        return self.observe_all_complete(indexes=indexes, num_features=1)




if __name__ == '__main__':
    import setters

    from modules.arguments import set_args

    args_dict = set_args()

    env = setters.EnvSetters.set_environment(args_dict=args_dict, id=0)
    observer = setters.ObserverSetters.set_observer(args_dict=args_dict, metaAgentID=0, env=env)

    env.reset()
    while True:
        actions = {}
        for id in env.agent_index:
            actions[id] = np.random.randint(0, env.TlsDict[id].action_space_n)

        _, reward, done, info = env.step(actions)
        observer_obs = observer.observe_all()
        obs = env.get_observations()
        print(obs)
