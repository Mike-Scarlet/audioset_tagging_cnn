
import h5py
import numpy as np

class TrainSampler:
  def __init__(self, batch_size, random_seed=1234):
    """Balanced sampler. Generate batch meta for training.
    
    Args:
      indexes_hdf5_path: string
      batch_size: int
      black_list_csv: string
      random_seed: int
    """
    self.local_path = r"J:\A10\manual_scripts\random\[[sound_dataset\example\out.h5"
    self.random_state = np.random.RandomState(random_seed)
    self.batch_size = batch_size

    # load audios num
    with h5py.File(self.local_path, 'r') as hf:
      (self.audios_num, self.classes_num) = hf['target'].shape
    
    self.indexes = np.arange(self.audios_num)
      
    # Shuffle indexes
    self.random_state.shuffle(self.indexes)
    
    self.pointer = 0

  def __iter__(self):
    """Generate batch meta for training. 
    
    Returns:
      batch_meta: e.g.: [
      {'hdf5_path': string, 'index_in_hdf5': int}, 
      ...]
    """
    batch_size = self.batch_size

    while True:
      indices = []
      i = 0
      while i < batch_size:
        index = self.indexes[self.pointer]
        self.pointer += 1

        # Shuffle indexes and reset pointer
        if self.pointer >= self.audios_num:
          self.pointer = 0
          self.random_state.shuffle(self.indexes)
        
        indices.append(index)
        i += 1

      yield indices

  def state_dict(self):
    state = {
      'indexes': self.indexes,
      'pointer': self.pointer}
    return state
      
  def load_state_dict(self, state):
    self.indexes = state['indexes']
    self.pointer = state['pointer']

# class BalancedTrainSampler:
#   def __init__(self, indexes_hdf5_path, batch_size, black_list_csv=None, 
#     random_seed=1234):
#     """Balanced sampler. Generate batch meta for training. Data are equally 
#     sampled from different sound classes.
    
#     Args:
#       indexes_hdf5_path: string
#       batch_size: int
#       black_list_csv: string
#       random_seed: int
#     """
#     self.local_path = r"J:\A10\manual_scripts\random\[[sound_dataset\example\out.h5"
#     self.random_state = np.random.RandomState(random_seed)
#     self.batch_size = batch_size

#     # load audios num
#     with h5py.File(self.local_path, 'r') as hf:
#       (self.audios_num, self.classes_num) = hf['target'].shape
    
#     self.samples_num_per_class = np.sum(self.targets, axis=0)
#     logging.info('samples_num_per_class: {}'.format(
#       self.samples_num_per_class.astype(np.int32)))
    
#     # Training indexes of all sound classes. E.g.: 
#     # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
#     self.indexes_per_class = []
    
#     for k in range(self.classes_num):
#       self.indexes_per_class.append(
#         np.where(self.targets[:, k] == 1)[0])
      
#     # Shuffle indexes
#     for k in range(self.classes_num):
#       self.random_state.shuffle(self.indexes_per_class[k])
    
#     self.queue = []
#     self.pointers_of_classes = [0] * self.classes_num

#   def expand_queue(self, queue):
#     classes_set = np.arange(self.classes_num).tolist()
#     self.random_state.shuffle(classes_set)
#     queue += classes_set
#     return queue

#   def __iter__(self):
#     """Generate batch meta for training. 
    
#     Returns:
#       batch_meta: e.g.: [
#       {'hdf5_path': string, 'index_in_hdf5': int}, 
#       ...]
#     """
#     batch_size = self.batch_size

#     while True:
#       batch_meta = []
#       i = 0
#       while i < batch_size:
#         if len(self.queue) == 0:
#           self.queue = self.expand_queue(self.queue)

#         class_id = self.queue.pop(0)
#         pointer = self.pointers_of_classes[class_id]
#         self.pointers_of_classes[class_id] += 1
#         index = self.indexes_per_class[class_id][pointer]
        
#         # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
#         if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
#           self.pointers_of_classes[class_id] = 0
#           self.random_state.shuffle(self.indexes_per_class[class_id])

#         # If audio in black list then continue
#         if self.audio_names[index] in self.black_list_names:
#           continue
#         else:
#           batch_meta.append({
#             'hdf5_path': self.hdf5_paths[index], 
#             'index_in_hdf5': self.indexes_in_hdf5[index]})
#           i += 1

#       yield batch_meta

#   def state_dict(self):
#     state = {
#       'indexes_per_class': self.indexes_per_class, 
#       'queue': self.queue, 
#       'pointers_of_classes': self.pointers_of_classes}
#     return state
      
#   def load_state_dict(self, state):
#     self.indexes_per_class = state['indexes_per_class']
#     self.queue = state['queue']
#     self.pointers_of_classes = state['pointers_of_classes']

class LocalH5Dataset(object):
  def __init__(self):
    """This class takes the meta of an audio clip as input, and return 
    the waveform and target of the audio clip. This class is used by DataLoader. 
    """
    self.local_path = r"J:\A10\manual_scripts\random\[[sound_dataset\example\out.h5"
  
  def __getitem__(self, index_in_hdf5):
    """Load waveform and target of an audio clip.
    
    Args:
      meta: {
      'hdf5_path': str, 
      'index_in_hdf5': int}

    Returns: 
      data_dict: {
      'audio_name': str, 
      'waveform': (clip_samples,), 
      'target': (classes_num,)}
    """
    with h5py.File(self.local_path, 'r') as hf:
      mel_feature = hf['mel_feature'][index_in_hdf5]
      target = hf['target'][index_in_hdf5].astype(np.float32)

    data_dict = {
      'index_in_hdf5': index_in_hdf5, 
      'mel_feature': mel_feature, 
      'target': target}
      
    return data_dict

def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict