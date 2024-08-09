# kubeflow(v1.8)-kserve 제공 torchserve image
FROM pytorch/torchserve-kfs:0.8.2

USER root
RUN pip install -U albumentations pydantic==1.10.11
RUN pip uninstall -y numpy
RUN pip install numpy==1.26.4

##  ipex :
#RUN python3 -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu \
#    && python3 -m pip install intel-extension-for-pytorch \
#    && python3 -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

USER model-server
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]