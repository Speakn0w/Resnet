import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "../RubbishClassification"
TRAINJSON_PATH = "../RubbishClassification/RubbishClassification/train.json"
TESTJSON_PATH = "../RubbishClassification/RubbishClassification/val.json"
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 1
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PATH = './model/resnet34-333f7ec4.pth'
TRAIN_ACC_PATH = ''
TEST_ACC_PATH = ''
LOSS_PATH = ''


transforms = A.Compose(
    [
        A.Resize(width=224, height=224),
        A.HorizontalFlip(p=0.5),


        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
)

# transforms2 = A.Compose(
#     [
#         A.OneOf([
#             A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
#             A.GaussNoise(),    # 将高斯噪声应用于输入图像。
#         ], p=0.2),   # 应用选定变换的概率
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
#         # 随机应用仿射变换：平移，缩放和旋转输入
#         A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
#         A.Resize(width=224, height=224),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
#         ToTensorV2(),
#      ],
# )