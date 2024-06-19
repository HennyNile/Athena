import random
import numpy as np

class QuerySampler:
    def __init__(self, dataset: list[list[dict]]):
        num_queries = len(dataset)
        self.all_indices = []
        sample_idx = 0
        for query in dataset:
            self.all_indices.append([sample_idx + i for i in range(len(query))])
            sample_idx += len(query)
        self.current_qid_idx = None

    def __len__(self):
        return len(self.all_indices)

    def __iter__(self):
        self.current_qid_idx = 0
        return self

    def __next__(self):
        if self.current_qid_idx >= len(self.all_indices):
            raise StopIteration
        ret = self.all_indices[self.current_qid_idx]
        self.current_qid_idx += 1
        return ret

class BatchedQuerySampler:
    def __init__(self, dataset: list[list[dict]], batch_size: int):
        self.batch_size = batch_size
        num_queries = len(dataset)
        self.all_indices = []
        sample_idx = 0
        for query in dataset:
            self.all_indices.extend([sample_idx + i for i in range(len(query))])
            sample_idx += len(query)
        self.num_plans = [len(query) for query in dataset]
        self.num_batches = (len(self.all_indices) + batch_size - 1) // batch_size
        self.current_batch = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        ret = self.all_indices[self.current_batch * self.batch_size: (self.current_batch + 1) * self.batch_size]
        self.current_batch += 1
        return ret
    
    def group(self, array: np.ndarray) -> list[np.ndarray]:
        ret = []
        idx = 0
        for num in self.num_plans:
            ret.append(array[idx:idx + num])
            idx += num
        return ret

class ItemwiseSampler:
    def __init__(self, dataset: list[list[dict]], batch_size: int, shuffle = True, drop_last = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.all_indices = []
        sample_idx = 0
        for query in dataset:
            for i in range(len(query)):
                self.all_indices.append(sample_idx + i)
            sample_idx += len(query)
        if not drop_last:
            self.num_batches = (len(self.all_indices) + batch_size - 1) // batch_size
        else:
            self.num_batches = len(self.all_indices) // batch_size
        self.shuffled_indices = None
        self.current_batch = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.shuffled_indices = self.all_indices[:]
        if self.shuffle:
            np.random.shuffle(self.shuffled_indices)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        ret = self.shuffled_indices[self.current_batch * self.batch_size: (self.current_batch + 1) * self.batch_size]
        self.current_batch += 1
        return ret

class PairwiseSampler:
    def __init__(self, dataset: list[list[dict]], batch_size: int, shuffle = True, drop_last = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.all_indices = []
        sample_idx = 0
        for query in dataset:
            for i in range(len(query) - 1):
                for j in range(i + 1, len(query)):
                    if 'Execution Time' in query[i] or 'Execution Time' in query[j]:
                        self.all_indices.append((sample_idx + i, sample_idx + j))
            sample_idx += len(query)
        if not drop_last:
            self.num_batches = (len(self.all_indices) + batch_size - 1) // batch_size
        else:
            self.num_batches = len(self.all_indices) // batch_size
        self.shuffled_indices = None
        self.current_batch = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.shuffled_indices = self.all_indices[:]
        if self.shuffle:
            np.random.shuffle(self.shuffled_indices)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        ret = self.shuffled_indices[self.current_batch * self.batch_size: (self.current_batch + 1) * self.batch_size]
        self.current_batch += 1
        return [idx for pair in ret for idx in pair]

class GroupState:
    def __init__(self, source, num_plans, shuffle):
        self.source = source
        self.shuffle = shuffle
        self.shuffled = self.source[:]
        if self.shuffle:
            np.random.shuffle(self.shuffled)
        self.offset = 0
        self.num_plans = num_plans

    def get_plans(self):
        ret = []
        while len(ret) < self.num_plans:
            n_plans = min(self.num_plans - len(ret), len(self.shuffled) - self.offset)
            if n_plans == 0:
                self.shuffled = self.source[:]
                if self.shuffle:
                    np.random.shuffle(self.shuffled)
                self.offset = 0
                n_plans = min(self.num_plans - len(ret), len(self.shuffled))
            ret.extend(self.shuffled[self.offset:self.offset + n_plans])
            self.offset += n_plans
        if len(ret) != self.num_plans:
            raise RuntimeError('Wrong implementation')
        return ret

class BalancedPairwiseSampler:
    def __init__(self, dataset: list[list[dict]], batch_size, shuffle = True, drop_last = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.states = []
        sample_idx = 0
        for query in dataset:
            indices = []
            for i in range(len(query) - 1):
                for j in range(i + 1, len(query)):
                    if 'Execution Time' in query[i] or 'Execution Time' in query[j]:
                        indices.append((sample_idx + i, sample_idx + j))
            num_finished_sample = sum([1 if 'Execution Time' in plan else 0 for plan in query])
            self.states.append(GroupState(indices, num_finished_sample, shuffle))
            sample_idx += num_finished_sample
        self.total_len = sample_idx
        if not drop_last:
            self.num_batches = (self.total_len + batch_size - 1) // batch_size
        else:
            self.num_batches = self.total_len // batch_size
        self.shuffled_indices = None
        self.current_batch = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.shuffled_indices = []
        for state in self.states:
            self.shuffled_indices.extend(state.get_plans())
        if self.shuffle:
            np.random.shuffle(self.shuffled_indices)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        ret = self.shuffled_indices[self.current_batch * self.batch_size: (self.current_batch + 1) * self.batch_size]
        self.current_batch += 1
        return [idx for pair in ret for idx in pair]