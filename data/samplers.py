import copy
import math
import random
import numpy as np
from torch import distributed as dist
from collections import defaultdict
from torch.utils.data.sampler import Sampler


def _get_dist_info(num_replicas=None, rank=None):
    """
    Safe distributed info getter.
    - If torch.distributed is not available or not initialized, fallback to (world_size=1, rank=0)
    - Allows manual override by passing num_replicas/rank explicitly.
    """
    if num_replicas is not None and rank is not None:
        return int(num_replicas), int(rank)

    if not dist.is_available() or not dist.is_initialized():
        # single-process fallback
        world_size, rank0 = 1, 0
        if num_replicas is None:
            num_replicas = world_size
        if rank is None:
            rank = rank0
        return int(num_replicas), int(rank)

    # distributed is initialized
    if num_replicas is None:
        num_replicas = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()
    return int(num_replicas), int(rank)


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source,batch_size, num_instances):
        self.batch_size = int(batch_size)
        self.data_source = data_source
        self.num_instances = num_instances
        assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            pid = item[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        list_container = []
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = list(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = idxs + random.choices(idxs, k=self.num_instances - len(idxs))
            random.shuffle(idxs)
            batch_idxs_dict[pid] = [idxs[i:i+self.num_instances] for i in range(0, len(idxs), self.num_instances)]
        avai_pids = self.pids.copy()
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                final_idxs.extend(batch_idxs_dict[pid].pop(0))
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length
    

# class DistributedRandomIdentitySampler(Sampler):
#     def __init__(self, data_source, batch_size, num_instances=4,
#                  num_replicas=None, rank=None, seed=0):

#         num_replicas, rank = _get_dist_info(num_replicas=num_replicas, rank=rank)
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.seed = seed
#         self.epoch = 0

#         self.data_source = data_source
#         self.batch_size = int(batch_size)
#         self.num_instances = int(num_instances)
#         assert self.batch_size % self.num_instances == 0, \
#             f"batch_size({self.batch_size}) must be divisible by num_instances({self.num_instances})"

#         self.num_pids_per_batch = self.batch_size // self.num_instances

#         self.index_dic = defaultdict(list)
#         for index, item in enumerate(data_source):
#             pid = item[1]
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())

#         # ---- 预估每个 epoch 能组成多少个 “K组” ----
#         self.num_groups = 0
#         for pid in self.pids:
#             n = len(self.index_dic[pid])
#             if n < self.num_instances:
#                 n = self.num_instances
#             n = n - n % self.num_instances
#             self.num_groups += n // self.num_instances

#         # 每个 batch 消耗 P 个 group
#         self.num_batches = self.num_groups // self.num_pids_per_batch
#         # DDP 需要 batch 数能整除 world_size，否则 rank 间步数不一致容易出问题
#         self.num_batches = self.num_batches - (self.num_batches % self.num_replicas)

#         self.num_samples = (self.num_batches // self.num_replicas) * self.batch_size
#         self.total_size = self.num_batches * self.batch_size

#     def __iter__(self):
#         random.seed(self.seed + self.epoch)
#         np.random.seed(self.seed + self.epoch)

#         P = self.num_pids_per_batch      # = batch_size // num_instances
#         K = self.num_instances
#         all_pids = self.pids

#         final_batches = []
#         for _ in range(self.num_batches):
#             # batch 内 pid 不重复：优先无放回抽样
#             if len(all_pids) >= P:
#                 selected_pids = random.sample(all_pids, P)
#             else:
#                 # 极端情况：pid 数不足（很少见），允许重复但仍尽量不重复
#                 selected_pids = list(np.random.choice(all_pids, size=P, replace=True))

#             batch = []
#             for pid in selected_pids:
#                 idxs = self.index_dic[pid]

#                 # 从该 pid 取 K 张
#                 if len(idxs) >= K:
#                     t = random.sample(idxs, K)          # 无放回取 K
#                 else:
#                     t = np.random.choice(idxs, size=K, replace=True).tolist()
#                 batch.extend(t)

#             final_batches.append(batch)

#         # 保证 batch 数能被 world_size 整除（你 nproc=1 时无所谓，多卡时必须）
#         num_batches = len(final_batches)
#         num_batches = num_batches - (num_batches % self.num_replicas)
#         final_batches = final_batches[:num_batches]

#         # rank 切分
#         final_batches = final_batches[self.rank:num_batches:self.num_replicas]

#         # flatten 成 indices
#         indices = []
#         for b in final_batches:
#             indices.extend(b)
#         return iter(indices)

#     def __len__(self):
#         return self.num_samples

#     def set_epoch(self, epoch):
#         self.epoch = epoch


class DistributedRandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    # Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    - num_replicas (int, optional): Number of processes participating in
        distributed training. By default, :attr:`world_size` is retrieved from the
        current distributed group.
    - rank (int, optional): Rank of the current process within :attr:`num_replicas`.
        By default, :attr:`rank` is retrieved from the current distributed group.
    - seed (int, optional): random seed used to shuffle the sampler. 
        This number should be identical across all
        processes in the distributed group. Default: ``0``.
    """
    def __init__(self, data_source, num_instances=4, 
                 num_replicas=None, rank=None, seed=0):
        # if num_replicas is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     num_replicas = dist.get_world_size()
        # if rank is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     rank = dist.get_rank()
        num_replicas, rank = _get_dist_info(num_replicas=num_replicas, rank=rank)
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        # for index, (_, pid, _, _) in enumerate(data_source):
        # for index, (_, pid, _, _,_) in enumerate(data_source):
        #     self.index_dic[pid].append(index)
        for index, item in enumerate(data_source):
            # item could be (img_path, pid, camid, clothid) or (img_path, pid, camid, clothid, aux_info)
            pid = item[1]
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        assert self.length % self.num_instances == 0

        if self.length // self.num_instances % self.num_replicas != 0: 
            self.num_samples = math.ceil((self.length // self.num_instances - self.num_replicas) / self.num_replicas) * self.num_instances
        else:
            self.num_samples = math.ceil(self.length / self.num_replicas) 
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)

        list_container = []
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    list_container.append(batch_idxs)
                    batch_idxs = []
        random.shuffle(list_container)

        # remove tail of data to make it evenly divisible.
        list_container = list_container[:self.total_size//self.num_instances]
        assert len(list_container) == self.total_size//self.num_instances

        # subsample
        list_container = list_container[self.rank:self.total_size//self.num_instances:self.num_replicas]
        assert len(list_container) == self.num_samples//self.num_instances

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. This ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedInferenceSampler(Sampler):
    """
    refer to: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py

    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, rank=None, num_replicas=None):
        # if num_replicas is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     num_replicas = dist.get_world_size()
        # if rank is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     rank = dist.get_rank()
        num_replicas, rank = _get_dist_info(num_replicas=num_replicas, rank=rank)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples