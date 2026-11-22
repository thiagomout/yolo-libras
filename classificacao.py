"""Transformações que preservam os dedos, sem recortes aleatórios da mão."""
from copy import copy

from torchvision import transforms as T
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator, ClassificationPredictor


def full_image_transforms(size: int, augment: bool = False):
    transforms = [T.Resize((size, size))]
    if augment:
        transforms += [
            T.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        ]
    return T.Compose(transforms + [T.ToTensor(), T.Normalize((0, 0, 0), (1, 1, 1))])


class FullImageValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        dataset = super().build_dataset(img_path)
        dataset.torch_transforms = full_image_transforms(self.args.imgsz)
        return dataset


class FullImageTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.torch_transforms = full_image_transforms(self.args.imgsz, mode == 'train')
        return dataset

    def get_validator(self):
        return FullImageValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


class FullImagePredictor(ClassificationPredictor):
    def setup_source(self, source):
        super().setup_source(source)
        self.transforms = full_image_transforms(max(self.imgsz))
        self.host_transforms = None
        self.device_transform = None
