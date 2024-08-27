import os

from yolox.exp import Exp as MyExp
import torch

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 4  # Number of classes in your dataset
        self.depth = 1.33
        self.width = 1.25
        self.data_dir = "/storage/datasets_public/RDD_YOLOV9/yolov9/datasets/All"
        self.train_ann = "images/train"
        self.val_ann = "images/val"
        self.input_size = (640, 640)  # Adjust if necessary
        self.test_size = (640, 640)  # Adjust if necessary
        self.basic_lr_per_img = 0.0001 # Set learning rate per image

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import get_yolox_datadir
        from yolox.data.datasets import COCODataset, TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection, worker_init_reset_seed

        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann if is_distributed else self.train_ann,
            name='train2017',
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
            cache=cache_img,
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=0.5, hsv_prob=1.0),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "worker_init_fn": worker_init_reset_seed,
        }

        train_loader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            **dataloader_kwargs,
        )

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import get_yolox_datadir
        from yolox.data.datasets import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name='val2017',
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        val_loader = torch.utils.data.DataLoader(
            valdataset,
            batch_size=batch_size,
            num_workers=self.data_num_workers,
            pin_memory=True,
            shuffle=False,
        )

        return val_loader

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            # Set learning rate here
            lr = self.basic_lr_per_img * batch_size
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True,
            )

        return self.optimizer
