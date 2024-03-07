1. data_preparation.py: mj synthetic dataset에는 900만개의 image data가 들어있어 용량이 너무 크기에 
train 5만개, validation 5천개, test 5천개를 추려내기 위해 data_preparation.py를 통해 미리 필요한 데이터 양 만큼만 뽑아내었다. 
제출 자료에는 총 6만(5만+5천+5천)개의 데이터만 포함.
csv 파일에는 각각 필요한 라벨링이 들어있다.

2. analysis.py: 입력 데이터의 resizing을 위해 필요한 데이터 사이즈 분석을 수행하는 프로그램. 이미지 넓이와 높이에 대한 분포 분석결과를 출력해 줌.

3. graph.py: 필요한 모델을 찾아내기 위해 구현한 성능 시각화 프로그램. 비교한 각 모델의 running time, accuracy, loss 값 비교를 위해 matplot 라이브러리를 통해
그래프를 출력하여 가장 우수한 모델을 찾을 수 있게 되었음.

4. {util_function.py, crnn.py, data_manager.py, run.py}: 프로젝트에 필요한 머신러닝 모델을 구현한 프로그램.
util_function.py에는 필요한 기능들이 들어있다.(이미지 resizing함수, predict output을 정의된 character로 변환해주는 함수 등)
data_manager.py에는 배치생성기와 데이터셋 불러오는 함수등이 정의되어 있다.
crnn.py에는 image text recognize에 필요한 crnn 모델이 구현되어 있다.
run.py에서는 argument를 파싱하여 train 혹은 test를 원하는 hyper parameter를 지정하여 실행할 수 있다.

5. 환경설정: requirements.txt를 통해 패키지 설치가 필요함.