import numpy as np
import sys
import os
import tensorflow as tf
from tslearn.generators import random_walk_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# shapelets.py 파일이 있는 폴더를 Python 경로에 추가
shapelets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shapelets"))
sys.path.append(shapelets_dir)

from shapelets import LearningShapelets

# Step 1: 로그 파일로 출력 저장 설정
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "test_output.txt"))

# 로그 디렉토리 생성
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# 표준 출력을 파일로 리디렉션
with open(log_file_path, "w", encoding="utf-8") as log_file:
    sys.stdout = log_file

    # TensorFlow Eager Execution 활성화
    tf.config.run_functions_eagerly(True)

    # Step 2: 데이터셋 생성
    print("Step 2: 데이터셋 생성...")
    X, y = random_walk_blobs(n_ts_per_blob=30, sz=100, d=5, n_blobs=3)

    # Step 3: 데이터 3D로 변환
    print("Step 3: 데이터셋 확인 및 변환...")
    X_reshaped = X.reshape(X.shape[0], X.shape[1], -1)
    y_reshaped = np.array(y)

    # Step 4: 학습/검증 데이터 분리
    print("Step 4: 학습/검증 데이터 분리...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_reshaped, test_size=0.2, random_state=42, stratify=y
    )

    # 입력 데이터를 float32로 변환
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    
    # Step 5: 모델 생성 및 학습
    print("Step 5: 모델 생성 및 설정...")
    shapelet_model = LearningShapelets(
        n_shapelets_per_size={10: 5, 20: 3},
        max_iter=500,
        batch_size=128,
        verbose=1
    )

    print("Step 6: 모델 학습 시작...")
    shapelet_model.fit(X_train, y_train)

    # Step 7: 테스트 데이터 예측 및 평가
    print("Step 7: 테스트 데이터 예측 및 평가...")
    y_pred = shapelet_model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Step 8: 결과 로그 저장 완료.")

# 표준 출력을 다시 콘솔로 복원
sys.stdout = sys.__stdout__

print(f"로그가 저장되었습니다: {log_file_path}")