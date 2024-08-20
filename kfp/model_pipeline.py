import json

from kfp.dsl import component, Input, Output, Artifact, Dataset

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
            raise Exception(f"{types} 공종이 존재 하지 않음.")

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
def data_prepare(input_data: Input(Artifact), output_data: Output[Dataset]):
    import pandas as pd
    import json

    with open(input_data.path, 'r') as f:
        images_path = json.load(f)

    data = pd.DataFrame(columns=['path', 'label'])
    for label, path_list in images_path.items():
        temp_df = pd.DataFrame({
            'path': path_list,
            'label': label
        })
        data = pd.concat([data, temp_df], ignore_index=True)

    data.to_csv(output_data.path, index=False)


def data_split(input_data: Input(Dataset)):
    print(input_data)


def model_train():
    pass

def model_validation():
    pass

def model_deploy():
    pass


def main():
    data_download_from_s3()


if __name__ == "__main__":
    main()
