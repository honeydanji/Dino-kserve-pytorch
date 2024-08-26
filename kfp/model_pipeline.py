import json

from kfp.dsl import component, Input, Output, Artifact, Dataset
CFG = {
  'IMG_SIZE': 520,
  'EPOCHS':300,
  'LEARNING_RATE':1e-3,
  'BATCH_SIZE':64, # 논문은 2048
  'SEED': 23,
  'feat_dim' : 384
}

@component(
    packages_to_install=["boto3"],
    base_image="python:3.9"
)
def data_download_from_s3(output_data: Output[Artifact]):
    import boto3
    import os
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("REGION")
    s3_kfp_client = boto3.client('s3',
                                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                 region_name=AWS_REGION)

    bucket_name = 'dino-test'
    start_after = 'kfp/images/'
    objects = s3_kfp_client.list_objects_v2(Bucket=bucket_name, StartAfter=start_after)['Contents']

    ## 모든 이미지를 호출하는 게 아니라 지정한 데이터 수에 도달한 공종만 호출해야 한다. (이후 수정이 필요함)
    images_path = {
        "wall": [],
        "tile": [],
        "pl": [],
        "etc": []
    }


    for obj in objects:
        key = obj['Key']
        types = key.split('/')[-2]

        if types not in images_path.keys():
            raise ValueError(f"{types} 공종이 존재 하지 않음.")

        if key.endswith(('.jpg', '.jpeg', '.png')):
            image_url = f'https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{key}'
            images_path[types].append(image_url)

            ## 이미지 원본 다운로드 (테스트용)
            # download_local_path = './images/'
            # folder_prefix = 'kfp/images/'

            # download_file_path = os.path.join(download_local_path, key[len(folder_prefix):])
            # os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
            # s3_kfp_client.download_file(bucket_name, key, download_file_path)
            # print(f"Downloaded {key} to {download_file_path}")

    with open(output_data.path, 'w') as f:
        json.dump(images_path, f)

@component(
    packages_to_install=["pandas"],
    base_image="python:3.9"
)
def data_preprocessing(input_data: Input(Artifact), output_data: Output[Dataset]):
    import pandas as pd
    import logging
    import json

    logging.basicConfig(level=logging.INFO)

    with open(input_data.path, 'r') as f:
        images_path = json.load(f)

    data = pd.DataFrame(columns=['path', 'label'])
    for label, path_list in images_path.items():
        temp_df = pd.DataFrame({
            'path': path_list,
            'label': label
        })
        data = pd.concat([data, temp_df], ignore_index=True)

    if data is None:
        logging.error("데이터가 존재 하지 않음.")
        raise ValueError("데이터 존재 하지 않음.")

    ## 데이터셋 라벨 종류 및 수 확인 - 공종 별 이미지 수 조건 걸기 가능.
    label_counts = data['label'].value_counts()
    for label, count in label_counts.items():
        print(f"{label}: {count}")
        logging.info(f"{label}: {count}")


    ## 데이터 csv 형식으로 저장
    data.to_csv(output_data.path, index=False)


@component(
    packages_to_install=["pandas", "scikit-learn", "albumentations"],
    base_image="python:3.9"
)
def data_split_and_post_processing(input_data: Input(Dataset)):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from albumentations.pytorch.transforms import ToTensorV2
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    import pandas as pd
    import numpy as np
    import cv2
    import requests

    encoder = LabelEncoder()

    data = pd.read_csv(input_data.path)

    ## 공종 추가에 따라 로직 변경
    data_df = pd.concat((
        data[data['label'] == 'wall'],
        data[data['label'] == 'tile'],
        data[data['label'] == 'pl'],
        data[data['label'] == 'etc']
    )).reset_index().drop('index', axis=1)

    ## 데이터 인코딩
    data_df['label'] = encoder.fit_transform(data['label'])
    data_df['path'] = data_df['path'].str.replace('\\', '/')

    ## 데이터 나누기
    X_train, X_valid, Y_train, Y_valid = (train_test_split(data_df['path'], data_df['label'], test_size=0.2,
                                                          stratify=data_df['label']))

    ## 트랜스폼 생성
    train_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.CenterCrop(518, 518),
        A.Rotate(p=0.5),
        A.ElasticTransform(p=0.5),
        A.OneOf([
            A.Downscale(),
            A.CLAHE(),
            A.Equalize(),
            A.GaussNoise(),
            A.ColorJitter(brightness=[0.9, 1.1], contrast=0.1, saturation=0.1, hue=0.01),
            A.CoarseDropout(max_holes=20, max_height=3, max_width=3, min_holes=1, min_height=1, min_width=1),
        ], p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                    always_apply=False, p=1.0),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.CenterCrop(518, 518),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    ## 커스텀 데이터셋 생성
    class CustomDataset(Dataset):
        def __init__(self, img_path_list, label_list, transforms=None):
            self.img_path_list = img_path_list
            self.label_list = label_list
            self.transforms = transforms

        def __getitem__(self, index):
            img_path = self.img_path_list[index]

            if img_path.startswith(('http', 'https')):
                response = requests.get(img_path)
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                image = np.fromfile(img_path, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if self.transforms is not None:
                image = self.transforms(image=image)['image']

            if self.label_list is not None:
                label = self.label_list[index]
                return image, label
            else:
                return image

        def __len__(self):
            return len(self.img_path_list)

    train_dataset = CustomDataset(X_train.values, Y_train.values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=64)

    val_dataset = CustomDataset(X_valid.values, Y_valid.values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=64)

    return train_loader, val_loader


def model_train():
    pass


def model_validation():
    pass


def model_update_to_s3():
    pass


def main():
    data_download_from_s3()


if __name__ == "__main__":
    main()
