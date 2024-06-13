import random

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
            random.shuffle(self.shuffled_indices)
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
            random.shuffle(self.shuffled_indices)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        ret = self.shuffled_indices[self.current_batch * self.batch_size: (self.current_batch + 1) * self.batch_size]
        self.current_batch += 1
        return [idx for pair in ret for idx in pair]