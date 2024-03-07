import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np


softplus6_loss_values = []
softplus6_accuracy_values = []
softplus6_time_values = []

file = open("훈련 layer6 softplus.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    softplus6_loss_values.append(loss)
    softplus6_accuracy_values.append(accuracy)
    softplus6_time_values.append(time)

file.close()


relu6_loss_values = []
relu6_accuracy_values = []
relu6_time_values = []

file = open("훈련 layer6 relu.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    relu6_loss_values.append(loss)
    relu6_accuracy_values.append(accuracy)
    relu6_time_values.append(time)

file.close()


leakyrelu6_loss_values = []
leakyrelu6_accuracy_values = []
leakyrelu6_time_values = []

file = open("훈련 layer6 leakyrelu.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    leakyrelu6_loss_values.append(loss)
    leakyrelu6_accuracy_values.append(accuracy)
    leakyrelu6_time_values.append(time)

file.close()


softplus7_loss_values = []
softplus7_accuracy_values = []
softplus7_time_values = []

file = open("훈련 layer7 softplus miw 170.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    softplus7_loss_values.append(loss)
    softplus7_accuracy_values.append(accuracy)
    softplus7_time_values.append(time)

file.close()


relu7_loss_values = []
relu7_accuracy_values = []
relu7_time_values = []

file = open("훈련 layer7 relu miw 170.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    relu7_loss_values.append(loss)
    relu7_accuracy_values.append(accuracy)
    relu7_time_values.append(time)

file.close()


leakyrelu7_loss_values = []
leakyrelu7_accuracy_values = []
leakyrelu7_time_values = []

file = open("훈련 layer7 leakyrelu miw 170.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    leakyrelu7_loss_values.append(loss)
    leakyrelu7_accuracy_values.append(accuracy)
    leakyrelu7_time_values.append(time)

file.close()


relu6_miw200_loss_values = []
relu6_miw200_accuracy_values = []
relu6_miw200_time_values = []

file = open("훈련 layer6 relu miw200.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    relu6_miw200_loss_values.append(loss)
    relu6_miw200_accuracy_values.append(accuracy)
    relu6_miw200_time_values.append(time)

file.close()


relu7_test_loss_values = []
relu7_test_accuracy_values = []
relu7_test_time_values = []

file = open("훈련 test data layer7 relu 모델.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    relu7_test_loss_values.append(loss)
    relu7_test_accuracy_values.append(accuracy)
    relu7_test_time_values.append(time)

file.close()


relu7_validation_loss_values = []
relu7_validation_accuracy_values = []
relu7_validation_time_values = []

file = open("훈련 validation data layer7 relu 모델.txt", "r", encoding = "UTF-8")

while True:
    line = file.readline()
    if not line:
        break
    line = line.strip()
    values = line.split()
    loss = values[1]
    accuracy = values[2]
    time = values[3]
    relu7_validation_loss_values.append(loss)
    relu7_validation_accuracy_values.append(accuracy)
    relu7_validation_time_values.append(time)

file.close()


x = np.linspace(0, 100, 100)

model1 = make_interp_spline(x, relu6_time_values)
y1 = model1(x)

model2 = make_interp_spline(x, leakyrelu6_time_values)
y2 = model2(x)

model3 = make_interp_spline(x, softplus6_time_values)
y3 = model3(x)

model4 = make_interp_spline(x, relu7_time_values)
y4 = model4(x)

model5 = make_interp_spline(x, leakyrelu7_time_values)
y5 = model5(x)

model6 = make_interp_spline(x, softplus7_time_values)
y6 = model6(x)

model7 = make_interp_spline(x, relu6_miw200_time_values)
y7 = model7(x)

model8 = make_interp_spline(x, relu7_test_time_values)
y8 = model8(x)

model9 = make_interp_spline(x, relu7_validation_time_values)
y9 = model9(x)

plt.figure(figsize=(18,9))

plt.plot(x, y1, color='blue', label='relu image_width 100')
# plt.plot(x, y2, color='red', label='laeckyrelu')
# plt.plot(x, y3, color='green', label='softplus')
# plt.plot(x, y4, color='blue', label='relu')
# plt.plot(x, y5, color='red', label='laeckyrelu')
# plt.plot(x, y6, color='green', label='softplus')
plt.plot(x, y7, color='green', label='relu image_width 200')
# plt.plot(x, y8, color='green', label='relu layer7 test')
# plt.plot(x, y9, color='blue', label='relu layer7 validation')

plt.legend()
plt.xlabel('iter')
plt.ylabel('run time')

plt.show()
