import numpy as np

class ReplayBuffer:
  """Abstract base class for replay buffer."""

  def __init__(self, data_spec, capacity=10000):
    """Initializes the replay buffer.

    Args:
      data_spec: A spec or a list/tuple/nest of specs describing
        a single item that can be stored in this buffer
      capacity: number of elements that the replay buffer can hold.
    """
    self.capacity = capacity
    self.occupied_capacity = 0
    self.buffer = {}
    self.buffer_keys = data_spec
    self.init_buffer()


  def init_buffer(self):
    """ Initialise the buffer"""
    for keys in self.buffer_keys:
        self.buffer[keys] = np.array([],dtype =np.object)

    self.occupied_capacity = 0

  def add_batch(self, items):
    """Adds a batch of items to the replay buffer.

    Args:
      items: Dictionary of list of transitions or trajectory buffer
    """
    trajectory_length = len(items[list(items.keys())[0]])
    if self.occupied_capacity + trajectory_length <= self.capacity:
        for key in items.keys():
            if key in self.buffer_keys:
                self.buffer[key] =np.concatenate((self.buffer[key],\
                                    np.array(items[key],dtype = np.object)),axis=0)
        self.occupied_capacity += trajectory_length

    else:

        shuffled_indices = np.arange(trajectory_length)
        np.random.shuffle(shuffled_indices)

        append_index = self.capacity - self.occupied_capacity

        # Sample with replacement
        shuffle_length = trajectory_length - append_index
        replaced_indices = np.random.choice(np.arange(self.capacity),shuffle_length,replace = False)
        self.occupied_capacity = self.capacity
        for key in items.keys():
            if key in self.buffer_keys:
                traj_partition_shuffled = np.array(items[key], dtype=np.object)[shuffled_indices]
                self.buffer[key] = np.concatenate((self.buffer[key],traj_partition_shuffled[:append_index]),axis=0)
                self.buffer[key][replaced_indices] = traj_partition_shuffled[append_index:]

  def get_next(self,sample_batch_size=None):
    """Returns an item or batch of items from the buffer."""
    if sample_batch_size>self.occupied_capacity:
        sample_indices = np.random.choice(np.arange(self.occupied_capacity), sample_batch_size, replace=True)
    else:
        sample_indices = np.random.choice(np.arange(self.occupied_capacity), sample_batch_size, replace=False)

    sampled_buffer = {}
    for key in self.buffer_keys:
        sampled_buffer[key] = self.buffer[key][sample_indices].tolist()

    return sampled_buffer

  def clear(self):
    """Resets the contents of replay buffer"""
    self.init_buffer()

class MACPGDynReplayBuffer(ReplayBuffer):
    def __init__(self,\
    data_spec  = ['observations,next_observations,actions,neighbours_actions'],\
    capacity = 10000):
        super(MACPGDynReplayBuffer, self).__init__(data_spec,capacity)


