import numpy as np
import sys
import os

# shapelets.py 파일이 있는 폴더를 Python 경로에 추가
shapelets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shapelets"))
sys.path.append(shapelets_dir)

from shapelets import LearningShapelets  # 수정된 import
from tslearn.generators import random_walk_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Step 1: 데이터셋 생성
# - 2개의 클래스가 있으며, 각 클래스에 50개의 시계열 데이터가 포함됨
X, y = random_walk_blobs(n_ts_per_blob=50, sz=100, d=5, n_blobs=2)

# Step 2: 3D로 변환 및 확인
# X는 이미 (instance, length, feature)의 형태이므로 reshape 필요 없음
X_reshaped = X.reshape(X.shape[0], X.shape[1], -1)  # 명시적으로 변형
y_reshaped = np.array(y)



# Step 3: 학습/검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_reshaped, test_size=0.2, random_state=42, stratify=y
)

# 4. 모델 생성 및 설정
shapelet_model = LearningShapelets(
    n_shapelets_per_size={10: 5, 20: 3},  # Shapelet 크기 설정
    max_iter=500,                       # 최대 반복 횟수
    batch_size=128,                      # 배치 사이즈                    # Z-정규화 확률                        
    verbose=1
)

# 5. 모델 학습
print("모델 학습 시작...")
shapelet_model.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = shapelet_model.predict(X_test)

# 정확도 및 평가 지표 출력
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")