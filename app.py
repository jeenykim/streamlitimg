import streamlit as st
# Streamlit 라이브러리를 st로 가져온다.
import tensorflow as tf
from PIL import Image, ImageOps
# PIL(Python Imaging Library) 라이브러리에서 Image와 ImageOps 모듈을 가져온다. 
# 이미지 조작 및 처리를 위한 기능을 제공한다.
import numpy as np
# NumPy 라이브러리를 가져온다. 
# 다차원 배열 및 행렬 연산을 지원한다.
from keras.applications.imagenet_utils import decode_predictions
# Keras의 decode_predictions 함수를 가져온다. 
# 이 함수는 이미지넷 데이터셋에 대한 예측 결과를 해석하는데 사용된다.
from keras.applications import ResNet50
# Keras의 ResNet50 모델을 가져온다. 
# 이 모델은 이미지 분류를 위해 사전 훈련된 ResNet50 아키텍처를 제공한다.


# pip install streamlit
# pip install tensorflow==2.15.0

# Streamlit을 사용하여 이미지를 업로드하고 
# ResNet50 모델을 사용하여 이미지를 분류하는 
# 간단한 이미지 분류 애플리케이션 
resnet50_pre =  tf.keras.applications.resnet.ResNet50(weights='imagenet' , input_shape=(224, 224, 3))
# ResNet50 모델을 가져와 resnet50_pre 변수에 할당한다. weights='imagenet'는 사전 훈련된 ImageNet 가중치를 사용하도록 지정하고, input_shape=(224, 224, 3)는 입력 이미지의 크기 및 채널 수를 지정한다.

st.title('이미지분류 인공지능입니다')
# Streamlit 애플리케이션의 타이틀을 설정한다.

file = st.file_uploader('이미지 올려주세요' , type=['jpg' , 'png'])
# 사용자로부터 이미지 파일을 업로드하기 위한 파일 업로더를 생성합니다. 허용되는 파일 형식은 JPEG와 PNG이다.

if file is None:
  st.text('이미지를 먼저 올려주세요')
  # 사용자가 이미지를 업로드하지 않았을 경우에는 메시지를 출력하고, 
else:
  image = Image.open(file)
  st.image(image, use_column_width=True)
  # 이미지를 업로드한 경우에는 해당 이미지를 출력한다.
  img_resized = ImageOps.fit(image, (224,224), Image.LANCZOS)
  # 업로드된 이미지를 ResNet50 모델에 입력할 수 있는 크기로 조정하고
  img_resized = img_resized.convert("RGB")
  # RGB 형식으로 변환한 후
  img_resized = np.asarray(img_resized)
  # NumPy 배열로 변환한다.
  
  pred = resnet50_pre.predict(img_resized.reshape([1, 224,224, 3]))
  # ResNet50 모델을 사용하여 이미지를 예측하고 예측 결과를 가져온다.
  decoded_pred = decode_predictions(pred)
  # 예측 결과를 해석하여 인간이 이해할 수 있는 형식으로 변환한다.
  
  results = ''
  for i, instance in enumerate(decoded_pred[0]):
    results += '{}위: {} {} ({:.2f}%) '.format(i+1, instance[1], instance[2], instance[2] * 100)
  st.success(results)
  # 예측 결과를 사용자에게 표시한다. 
  # 예측 결과에서 상위 몇 개의 클래스와 그에 대한 확률을 출력한다