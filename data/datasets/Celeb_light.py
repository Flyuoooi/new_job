# import os
# import re
# import glob
# import h5py
# import random
# import math
# import logging
# import numpy as np
# import os.path as osp
# from scipy.io import loadmat
# from tools.utils import mkdir_if_missing, write_json, read_json


# class Celeb_light(object):
#     """ Celeb-reID-light

#     Reference:
#         Huang et al. Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification. IJCNN, 2019.

#     URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
#     """
#     dataset_dir = 'Celeb-reID-light'
#     def __init__(self, root='data',aux_info =False, meta_dir='PAR_PETA_105.txt',meta_dims=105, **kwargs):
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.aux_info = aux_info
#         self.meta_dir = meta_dir
#         self.meta_dims = meta_dims
#         self.train_dir = osp.join(self.dataset_dir, 'train')
#         self.query_dir = osp.join(self.dataset_dir, 'query')
#         self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
#         self._check_before_run()

#         train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
#             self._process_dir_train(self.train_dir)
#         query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
#             self._process_dir_test(self.query_dir, self.gallery_dir)
#         num_total_pids = num_train_pids + num_test_pids
#         num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
#         num_test_imgs = num_query_imgs + num_gallery_imgs 
#         num_total_clothes = num_train_clothes + num_test_clothes

#         logger = logging.getLogger('reid.dataset')
#         logger.info("=> Celeb loaded")
#         logger.info("Dataset statistics:")
#         logger.info("  ----------------------------------------")
#         logger.info("  subset   | # ids | # images | # clothes")
#         logger.info("  ----------------------------------------")
#         logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
#         logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
#         logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
#         logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
#         logger.info("  ----------------------------------------")
#         logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
#         logger.info("  ----------------------------------------")

#         self.train = train
#         self.query = query
#         self.gallery = gallery

#         self.num_train_pids = num_train_pids
#         self.num_train_clothes = num_train_clothes  # 9021
#         self.num_test_clothes = num_test_clothes  # 1821
#         self.num_query_imgs = num_query_imgs
#         self.pid2clothes = pid2clothes

#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.query_dir):
#             raise RuntimeError("'{}' is not available".format(self.query_dir))
#         if not osp.exists(self.gallery_dir):
#             raise RuntimeError("'{}' is not available".format(self.gallery_dir))

#     def _process_dir_train(self, dir_path):
#         img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
#         img_paths.sort()
#         pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
#         pattern2 = re.compile(r'(\w+)_')

#         pid_container = set()
#         clothes_container = set()
#         for img_path in img_paths:
#             pid, _, _ = map(int, pattern1.search(img_path).groups())
#             clothes_id = pattern2.search(img_path).group(1)
#             pid_container.add(pid)
#             clothes_container.add(clothes_id)
#         imgdir2attribute = None
#         if self.aux_info:
#             imgdir2attribute = {}
#             with open(os.path.join(self.dataset_dir, self.meta_dir), 'r') as f:
#                 for line in f:
#                     imgdir, attribute_id, is_present = line.split()
#                     if imgdir not in imgdir2attribute:
#                         imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
#                     imgdir2attribute[imgdir][int(attribute_id)] = int(is_present)
#         pid_container = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         pid2label = {pid:label for label, pid in enumerate(pid_container)}
#         clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

#         num_pids = len(pid_container)
#         num_clothes = len(clothes_container)

#         dataset = []
#         images_info = []
#         pid2clothes = np.zeros((num_pids, num_clothes))
#         for img_path in img_paths:
#             pid, _, camid = map(int, pattern1.search(img_path).groups())
#             clothes = pattern2.search(img_path).group(1)
#             # camid -= 1 # index starts from 0
#             pid = pid2label[pid]
#             clothes_id = clothes2label[clothes]
#             if self.aux_info:
#                 dataset.append((img_path, pid, camid, clothes_id, imgdir2attribute[img_path]))
#             else:
#                 dataset.append((img_path, pid, camid, clothes_id))
#             pid2clothes[pid, clothes_id] = 1
        
#         num_imgs = len(dataset)

#         return dataset, num_pids, num_imgs, num_clothes, pid2clothes

#     def _process_dir_test(self, query_path, gallery_path):
#         query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
#         gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
#         query_img_paths.sort()
#         gallery_img_paths.sort()
#         pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
#         pattern2 = re.compile(r'(\w+)_')

#         pid_container = set()
#         clothes_container = set()

#         for img_path in query_img_paths:
#             pid, _, _ = map(int, pattern1.search(img_path).groups())
#             clothes_id = pattern2.search(img_path).group(1)
#             pid_container.add(pid)
#             clothes_container.add(clothes_id)
#         for img_path in gallery_img_paths:
#             pid, _, _ = map(int, pattern1.search(img_path).groups())
#             clothes_id = pattern2.search(img_path).group(1)
#             pid_container.add(pid)
#             clothes_container.add(clothes_id)
#         pid_container = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         imgdir2attribute = {}
#         with open(os.path.join(self.dataset_dir, self.meta_dir), 'r') as f:
#             for line in f:
#                 imgdir, attribute_id, is_present = line.split()
#                 if imgdir not in imgdir2attribute:
#                     imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
#                 imgdir2attribute[imgdir][int(attribute_id)] = int(is_present)

#         pid2label = {pid:label for label, pid in enumerate(pid_container)}
#         clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

#         num_pids = len(pid_container)
#         num_clothes = len(clothes_container)

#         query_dataset = []
#         gallery_dataset = []
#         images_info_query = []
#         images_info_gallery = []
#         for img_path in query_img_paths:
#             pid, _, camid = map(int, pattern1.search(img_path).groups())
#             clothes_id = pattern2.search(img_path).group(1)
#             # camid -= 1 # index starts from 0
#             clothes_id = clothes2label[clothes_id]
#             if self.aux_info:
#                 query_dataset.append((img_path, pid, camid, clothes_id, imgdir2attribute[img_path]))
#             else:
#                 query_dataset.append((img_path, pid, camid, clothes_id))
#             images_info_query.append({'attributes': imgdir2attribute[img_path]})

#         for img_path in gallery_img_paths:
#             pid, _, camid = map(int, pattern1.search(img_path).groups())
#             clothes_id = pattern2.search(img_path).group(1)
#             # camid -= 1 # index starts from 0
#             clothes_id = clothes2label[clothes_id]
#             if self.aux_info:
#                 gallery_dataset.append((img_path, pid, camid, clothes_id, imgdir2attribute[img_path]))
#             else:
#                 gallery_dataset.append((img_path, pid, camid, clothes_id))
#             images_info_gallery.append({'attributes': imgdir2attribute[img_path]})
        
#         num_imgs_query = len(query_dataset)
#         num_imgs_gallery = len(gallery_dataset)

#         return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes

# if __name__ =='__main__':
#     dataset=Celeb_light('/data/Data/ReIDData')
#     print(dataset.num_train_clothes)
#     print(dataset.num_test_clothes)
import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json


class Celeb_light(object):
    """ Celeb-reID-light

    Reference:
        Huang et al. Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification. IJCNN, 2019.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'Celeb-reID-light'
    
    def __init__(self, root='data', aux_info=False, meta_dir='PAR_PETA_105.txt', meta_dims=105, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.aux_info = aux_info
        self.meta_dir = meta_dir
        self.meta_dims = meta_dims
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> Celeb loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes  # 9021
        self.num_test_clothes = num_test_clothes  # 1821
        self.num_query_imgs = num_query_imgs
        self.pid2clothes = pid2clothes

    # def _check_before_run(self):
    #     """Check if all files are available before going deeper"""
    #     if not osp.exists(self.dataset_dir):
    #         raise RuntimeError("'{}' is not available".format(self.dataset_dir))
    #     if not osp.exists(self.train_dir):
    #         raise RuntimeError("'{}' is not available".format(self.train_dir))
    #     if not osp.exists(self.query_dir):
    #         raise RuntimeError("'{}' is not available".format(self.query_dir))
    #     if not osp.exists(self.gallery_dir):
    #         raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

        # 只有 aux_info=True 才要求 meta 文件存在
        if self.aux_info:
            meta_path = os.path.join(self.dataset_dir, self.meta_dir)
            if not osp.exists(meta_path):
                raise RuntimeError(f"aux_info=True but meta file not found: {meta_path}")


    def _process_dir_train(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        
        # 修改1：初始化为空字典，并增加 aux_info 判断
        imgdir2attribute = {}
        if self.aux_info:
            with open(os.path.join(self.dataset_dir, self.meta_dir), 'r') as f:
                for line in f:
                    imgdir, attribute_id, is_present = line.split()
                    if imgdir not in imgdir2attribute:
                        imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
                    imgdir2attribute[imgdir][int(attribute_id)] = int(is_present)
        
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        # images_info = [] # 删除：未使用的变量，且会导致KeyError
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)
            # camid -= 1 # index starts from 0
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            
            # 修改2：根据 aux_info 决定是否读取属性
            if self.aux_info:
                dataset.append((img_path, pid, camid, clothes_id, imgdir2attribute[img_path]))
                # images_info.append({'attributes': imgdir2attribute[img_path]}) # 删除
            else:
                dataset.append((img_path, pid, camid, clothes_id))
            
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()

        pattern1 = re.compile(r'(\d+)_(\d+)_(\d+)')
        pattern2 = re.compile(r'(\w+)_')

        pid_container = set()
        clothes_container = set()

        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)

        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)

        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)

        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        # ✅ PRCC 对齐：aux_info 才读 meta；并对 meta_dims 越界保护
        imgdir2attribute = None
        if self.aux_info:
            imgdir2attribute = {}
            meta_path = os.path.join(self.dataset_dir, self.meta_dir)
            with open(meta_path, 'r') as f:
                for line in f:
                    imgdir, attribute_id, is_present = line.split()
                    aid = int(attribute_id)
                    if imgdir not in imgdir2attribute:
                        imgdir2attribute[imgdir] = [0 for _ in range(self.meta_dims)]
                    if aid >= self.meta_dims:
                        continue
                    imgdir2attribute[imgdir][aid] = int(is_present)

        query_dataset = []
        gallery_dataset = []

        # ✅ 关键修复：Celeb-reID-light 没有“真实多摄像头”意义上的 camid
        # Market1501-style eval 会过滤同 pid 同 cam 的正样本，导致 num_valid_q==0。
        # 解决：强制 query camid=0，gallery camid=1，避免正样本被 same-cam filter 全删光。
        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_key = pattern2.search(img_path).group(1)
            clothes_id = clothes2label[clothes_key]
            camid = 0  # force query camid
            if self.aux_info:
                query_dataset.append((img_path, pid, camid, clothes_id, imgdir2attribute[img_path]))
            else:
                query_dataset.append((img_path, pid, camid, clothes_id))

        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_key = pattern2.search(img_path).group(1)
            clothes_id = clothes2label[clothes_key]
            camid = 1  # force gallery camid
            if self.aux_info:
                gallery_dataset.append((img_path, pid, camid, clothes_id, imgdir2attribute[img_path]))
            else:
                gallery_dataset.append((img_path, pid, camid, clothes_id))

        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes

if __name__ =='__main__':
    dataset=Celeb_light('/data/Data/ReIDData')
    print(dataset.num_train_clothes)
    print(dataset.num_test_clothes)