import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import ResNet50
from googletrans import Translator

# pip install streamlit
# pip install tensorflow==2.15.0
# pip install googletrans==4.0.0-rc1

# Google 번역기 초기화
translator = Translator()

# 클래스 이름을 영어에서 한글로 번역하는 함수
def translate_to_korean(english_name):
    try:
        # 번역
        translation = translator.translate(english_name, src='en', dest='ko')
        return translation.text
    except Exception as e:
        st.error(f"번역 오류: {e}")
        return english_name  # 오류가 발생하면 영어 그대로 출력

# ResNet50 모델 로드
resnet50_pre = tf.keras.applications.resnet.ResNet50(weights='imagenet', input_shape=(224, 224, 3))

# Streamlit UI 설정
st.title('이미지 분류 인공지능')

# 사용자로부터 이미지 파일 업로드 받기
file = st.file_uploader('이미지 올려주세요', type=['jpg', 'png'])

if file is None:
    st.text('이미지를 먼저 올려주세요')
else:
    # 업로드된 이미지 열기
    image = Image.open(file)
    st.image(image)

    # 이미지 리사이즈 및 전처리
    img_resized = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    img_resized = img_resized.convert("RGB")
    img_resized = np.asarray(img_resized)

    # 예측 실행
    pred = resnet50_pre.predict(img_resized.reshape([1, 224, 224, 3]))
    decoded_pred = decode_predictions(pred)

    # 예측 결과를 한글로 변환하여 출력
    results = ''
    for i, instance in enumerate(decoded_pred[0]):
        english_name = instance[1]  # 예측된 영어 클래스 이름
        korean_name = translate_to_korean(english_name)  # 한글로 변환
        confidence = instance[2] * 100  # 신뢰도

        results += '{}위: {} ({:.2f}%) '.format(i + 1, korean_name, confidence)

    st.success(results)  # 결과 출력
