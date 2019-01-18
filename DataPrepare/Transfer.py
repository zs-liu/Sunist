import cv2
import random
import numpy as np
import torch


class Taker:

    def __init__(self, input_dir, output_dir, interval=1, random_take=False):
        print("New Transfer Prepared!")
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.interval = int(max(interval, 1))
        self.random_take = random
        self.cap = cv2.VideoCapture(input_dir)
        print("Video Loading Success!")
        self.number = 0
        self.height = 0
        self.width = 0

    def take(self):
        success, frame = self.cap.read()
        i = 0
        i_help = 0
        index = 0
        print(frame.shape)
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        while success:
            if i == i_help:
                cv2.imwrite(self.output_dir + str(index) + ".png", frame)
                if self.random_take:
                    i_help = i_help + random.randint(1, self.interval)
                else:
                    i_help = i_help + self.interval
                index = index + 1

            success, frame = self.cap.read()
            i = i + 1
        print("Transfer Finish!")
        self.number = index


class Processor:

    def __init__(self, input_dir, output_dir, number):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.number = number

    def grayer(self):
        for i in range(0, self.number):
            img = cv2.imread(self.input_dir + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(self.output_dir + str(i) + ".png", img)

        print("Gray Process Success!")


class Croper:

    def __init__(self, data_input_dir, label_input_dir, data_output_dir, label_output_dir, number):
        self.data_input_dir = data_input_dir
        self.label_input_dir = label_input_dir
        self.data_output_dir = data_output_dir
        self.label_output_dir = label_output_dir
        self.number = number

    def crop(self, x_pos, y_pos, crop_width, crop_height):
        for i in range(0, self.number):
            data_img = cv2.imread(self.data_input_dir + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
            label_img = cv2.imread(self.label_input_dir + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
            label_img = (label_img == 255) * 255

            random_crop = random.randint(0, 50 + i // 2)
            cv2.imwrite(self.data_output_dir + str(i) + ".png",
                        data_img[y_pos + random_crop:y_pos + crop_height + random_crop,
                        x_pos + random_crop:x_pos + crop_width + random_crop])
            cv2.imwrite(self.label_output_dir + str(i) + ".png",
                        label_img[y_pos + random_crop:y_pos + crop_height + random_crop,
                        x_pos + random_crop:x_pos + crop_width + random_crop])

            random_crop = random.randint(0, 50 + i // 2)
            cv2.imwrite(self.data_output_dir + str(i + self.number) + ".png",
                        data_img[y_pos + random_crop:y_pos + crop_height + random_crop,
                        x_pos + crop_width + random_crop:x_pos + random_crop:-1])
            cv2.imwrite(self.label_output_dir + str(i + self.number) + ".png",
                        label_img[y_pos + random_crop:y_pos + crop_height + random_crop,
                        x_pos + crop_width + random_crop:x_pos + random_crop:-1])


class Producer:

    def __init__(self, input_dir, output_dir):
        self.cap = cv2.VideoCapture(input_dir)
        print("Video Loading Success!")
        self.vwrite = cv2.VideoWriter(output_dir, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 20, (1280, 800))

    def read(self):
        return self.cap.read()

    def write(self, frame):
        self.vwrite.write(frame)

    def release(self):
        self.cap.release()
