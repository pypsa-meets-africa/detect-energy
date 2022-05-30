import os
# os.chdir('/disk3/fioriti/git/detect_energy')
# from dotenv import find_dotenv, load_dotenv
# load_dotenv(find_dotenv())

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

import sys
# sys.path.append(os.environ.get('PROJECT_ROOT'))
# data_path = os.environ.get('PROJECT_DATASETS')
# model_out_path = "/disk3/fioriti/git/detect_energy/cluster-runs/models/PISA_Parameter_Tuning"

model_out_path = "/content/drive/MyDrive/PyPSA_Africa_images/models/transfer/australia_matched"


from itertools import product
import json
from attrdict import AttrDict
from sklearn.model_selection import ParameterGrid

from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances 
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFilter
from typing import List, Union, Optional
import logging
import copy
import random

import detectron2.data.detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper

class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        
        augmentation.append(
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
                )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 1.5])], p=1))

        datatype_transform = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(),])
        augmentation.append(datatype_transform)
        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)


class DatasetMapperAugment(DatasetMapper):
    """
    This customized mapper produces two augmented images from a single image
    instance. This mapper makes sure that the two augmented images have the same
    cropping and thus the same size.
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.
    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)

        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format

        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(image_pil))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )
        return dataset_dict


DATASETS_PATH = '/content/drive/MyDrive/PyPSA_Africa_images/datasets'

print("DATASETS_PATH: " + DATASETS_PATH)

def register_all():
    # register used datasets
    ds_names = ['australia']
    modes = ['val']

    for name, mode in product(ds_names, modes):

        ds_name = f'{name}_{mode}'
        json_path = os.path.join(DATASETS_PATH, f'{ds_name}/labels.json')
        ds_path = os.path.join(DATASETS_PATH, f'{ds_name}/data/')

        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
            MetadataCatalog.remove(ds_name)

        register_coco_instances(ds_name, {}, json_path, ds_path)

    ds_name = 'transmission_04_train'
    json_path = os.path.join(DATASETS_PATH, f'{ds_name}/labels.json')
    ds_path = os.path.join(DATASETS_PATH, f'{ds_name}/data/')

    if ds_name in DatasetCatalog.list():
        DatasetCatalog.remove(ds_name)
        MetadataCatalog.remove(ds_name)

    register_coco_instances(ds_name, {}, json_path, ds_path)

register_all()


class TuneTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        if isinstance(cfg.DATASETS.EVAL, str):
            self.eval_datasets = [cfg.DATASETS.EVAL]
        else:
            self.eval_datasets = cfg.DATASETS.EVAL

        # prepare evaluation
        self.eval_loaders = []
        self.evaluators = []
        for dataset in self.eval_datasets:

            loader = build_detection_test_loader(DatasetCatalog.get(dataset), 
                                                 mapper=DatasetMapper(cfg, is_train=False))

            self.eval_loaders.append(loader)
            self.evaluators.append(COCOEvaluator(dataset))


    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True))



    def after_step(self):
        super().after_step()

        if (self.iter+1) % self.cfg.TEST.INTERVAL == 0:                                   

            for dataset, loader, evaluator in zip(self.eval_datasets, 
                                                  self.eval_loaders,
                                                  self.evaluators):

                results = inference_on_dataset(self.model,
                                              loader,
                                              evaluator)
                with open(
                    os.path.join(
                        self.cfg.OUTPUT_DIR,
                        'eval_'+dataset+'_iter_'+str(self.iter)+'.json'),
                        'w') as out:
                    json.dump(results, out)



def run_parameters(params):
    print(f'Starting run for parameters: {params}')
    params = AttrDict(params)

    cfg = get_cfg()

    # From Detectron2 Model Zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/" + params.model_type))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/" + params.model_type)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1

    cfg.DATASETS.TRAIN = 'transmission_04_train'
    cfg.DATASETS.TEST = ['manual_maxar_val']
    cfg.DATASETS.EVAL = ['australia_val']

    cfg.TEST.INTERVAL = 1000
    cfg.SOLVER.MAX_ITER = 12_000
    cfg.SOLVER.STEPS = (8_000, 10_000)

    # setup current parameters
    # cfg.SOLVER.IMS_PER_BATCH = params['SOLVER.IMS_PER_BATCH']
    # cfg.SOLVER.BASE_LR = params['SOLVER.BASE_LR']
    # cfg.SOLVER.MOMENTUM = params['SOLVER.MOMENTUM']
    # cfg.SOLVER.WEIGHT_DECAY = params['SOLVER.WEIGHT_DECAY']
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = params['MODEL.ANCHOR_GENERATOR.SIZES']

    # select_model = params.model_type.split(".")[0]

    model_name = f"model"
    # model_name +=f"LR_{cfg.SOLVER.BASE_LR}_"
    # model_name +=f"IMSPERBATCH_{cfg.SOLVER.IMS_PER_BATCH}_"
    # model_name +=f"MOM_{cfg.SOLVER.MOMENTUM}_"
    # model_name +=f"WEIGHTDECAY_{cfg.SOLVER.WEIGHT_DECAY}"
    # model_name +=f"ANCHORS"+ \
    #            str(cfg.MODEL.ANCHOR_GENERATOR.SIZES).replace('[[', '_').replace(']]', '_').replace(',', '_').replace(' ', '')

    cfg.OUTPUT_DIR = os.path.join(model_out_path, model_name)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = TuneTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    trainer.train()


if __name__ == '__main__':

    parameters = {
        'model_type': ['faster_rcnn_R_101_FPN_3x.yaml'],
        'SOLVER.BASE_LR': [1e-3],           # default
        'SOLVER.MOMENTUM': [0.9],           # default
        'SOLVER.IMS_PER_BATCH': [8],
        'SOLVER.WEIGHT_DECAY': [0.0001],    # first one is default
        'MODEL.ANCHOR_GENERATOR.SIZES': [[10, 20, 40, 80, 160]],
        }

    parameter_sweep = list(ParameterGrid(parameters))

    for params in parameter_sweep:
        run_parameters(params)