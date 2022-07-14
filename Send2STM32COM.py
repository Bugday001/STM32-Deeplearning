import serial
import time
from tensorflow.keras.datasets import mnist  # 数据集


def ReadImgBuffer():
    (_, _), (x_test, _) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1)
    with open("send_txt.txt", "w") as f:
        f.write("(")
        for data in x_test[6].reshape(28, 28):
            for index, each in enumerate(data):
                f.write(str(each) + ",")
            f.write("\n")
        f.write(")")


def Send2STM32_fromFile():
    ser = serial.Serial("COM4", 115200, timeout=0.5)
    with open("send_txt.txt", "r") as file:
        for line in file:
            ser.write(line.encode())
            time.sleep(0.1)
    print("ok!")
    ser.close()


def Send2STM32_fromKeras(num=0):
    ser = serial.Serial("COM4", 115200, timeout=0.5)
    (_, _), (x_test, _) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1)
    each_data = ""
    ser.write("(".encode())
    for data in x_test[num].reshape(28, 28):
        for index, each in enumerate(data):
            each_data += str(each) + ","
        ser.write(each_data.encode())
        each_data = ""
        time.sleep(0.2)
    ser.write(")".encode())
    print("ok!")
    ser.close()


if __name__ == "__main__":
    # Send2STM32_fromFile()
    Send2STM32_fromKeras(55)
    # for each in range(50, 51):
    #     Send2STM32_fromKeras(each)
    #     time.sleep(3)
