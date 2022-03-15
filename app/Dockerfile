FROM public.ecr.aws/lambda/python:3.8

RUN yum install libgomp git -y \
 && yum clean all -y && rm -rf /var/cache/yum

ARG TORCH_VER=1.9.1+cpu
ARG TORCH_VISION_VER=0.10.1+cpu
# ARG NUMPY_VER=1.21.5
ARG AUTOGLUON_VERSION='0.4.0'
RUN python3.8 -m pip --no-cache-dir install --upgrade --trusted-host pypi.org --trusted-host files.pythonhosted.org pip \
 && python3.8 -m pip --no-cache-dir install --upgrade wheel setuptools \
 && python3.8 -m pip uninstall -y dataclasses \
 && python3.8 -m pip --no-cache-dir install --upgrade torch=="${TORCH_VER}" torchvision=="${TORCH_VISION_VER}" -f https://download.pytorch.org/whl/torch_stable.html \
#  && python3.8 -m pip --no-cache-dir install --upgrade numpy==${NUMPY_VER} \
 && python3.8 -m pip --no-cache-dir install --upgrade autogluon.tabular[all]=="${AUTOGLUON_VERSION}"

COPY app.py ./
COPY ag_models /opt/ml/model/

CMD ["app.lambda_handler"]
