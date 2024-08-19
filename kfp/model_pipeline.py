from kfp import dsl
import boto3
import os


@dsl.component
def data_download_from_s3():
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

    images_for_train = []

    for obj in objects:
        key = obj['Key']
        if key.endswith(('.jpg', '.jpeg', '.png')):
            image_url = f'https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{key}'
            images_for_train.append(image_url)

            ## 이미지 원본 다운로드 (테스트용)
            # download_local_path = './images/'
            # folder_prefix = 'kfp/images/'

            # download_file_path = os.path.join(download_local_path, key[len(folder_prefix):])
            # os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
            # s3_kfp_client.download_file(bucket_name, key, download_file_path)
            # print(f"Downloaded {key} to {download_file_path}")

    print(images_for_train)
    return images_for_train


def data_prepare():
    pass


def data_split():
    pass


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
