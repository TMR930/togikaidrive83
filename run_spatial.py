#!/usr/bin/env python3
# coding: utf-8

import argparse
import cv2
import numpy as np
import sys
import threading
import time
import RPi.GPIO as GPIO
from typing import Tuple, TypedDict

from lib.oakd_spatial_yolo import OakdSpatialYolo

import config
from motor import Motor
import planner
import ultrasonic


GPIO.setwarnings(False)
# GPIOピン番号の指示方法
GPIO.setmode(GPIO.BOARD)
GPIO.setup(config.e_list, GPIO.IN)
GPIO.setup(config.t_list, GPIO.OUT, initial=GPIO.LOW)

# 以下はconfig.pyでの設定によりimport
if config.HAVE_CONTROLLER:
    from joystick import Joystick
if config.HAVE_NN:
    import train_pytorch


OFFSET_ARROW_X = 200  # 矢印看板から走行位置までのX座標オフセット(mm)
DETECTION_DISTANCE_LIMIT = 3000  # 一定距離以上の検出物をカット(mm)
PARKING_TIME = 500  # 一定時間経過したら駐車モードにする(sec)

detections = []  # 認識結果を格納しスレッド間で共有
parking_mode = False

# 引数設定
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    help="Provide model name or model path for inference",
    # default="yolov7tiny_coco_416x416",
    # default="models/aicar_20240825.blob",
    default="models/aicar_20240908.blob",
    type=str,
)
parser.add_argument(
    "-c",
    "--config",
    help="Provide config path for inference",
    # default="json/yolov7tiny_coco_416x416.json",
    # default="json/aicar_20240825.json",
    default="json/aicar_20240908.json",
    type=str,
)
parser.add_argument(
    "-f",
    "--fps",
    help="Camera frame fps. This should be smaller than nn inference fps",
    default=10,
    type=int,
)
parser.add_argument(
    "-s",
    "--save_fps",
    help="Image save fps. If it's > 0, images and video will be saved.",
    default=0,
    type=int,
)
args = parser.parse_args()


class Object(TypedDict):
    name: str = ""  # 物体ラベル
    id: int = None  # 最新のトラッキングID
    prev_status: bool = False  # 1つ前のstatus
    status: bool = False  # 現在認識しているかどうか
    pos: np.array = np.array([0.0, 0.0, 0.0])  # 位置
    start_time: float = 0  # 認識開始時刻
    last_time: float = 0  # 最後に認識した時刻
    prev_focus_status: bool = False  # 1つ前のfocus_status


def measure_ultrasonic(d, ultrasonics):
    # 認知（超音波センサ計測）
    # RrRHセンサ距離計測例：dis_RrRH = ultrasonic_RrRH.()
    message = ""
    for i, name in enumerate(config.ultrasonics_list):
        d[i] = ultrasonics[name].measure()
        message += name + ":" + \
            "{:>4}".format(round(ultrasonics[name].dis)) + ", "
    return message


def planning_ultrasonic(plan, ultrasonics, model):
    # 判断（プランニング）
    # 使う超音波センサをconfig.pyのultrasonics_listで設定必要
    # ただ真っすぐに走る
    if config.mode_plan == "GoStraight":
        steer_pwm_duty, throttle_pwm_duty = 0, config.FORWARD_S
    # 右左空いているほうに走る
    elif config.mode_plan == "Right_Left_3":
        steer_pwm_duty, throttle_pwm_duty = plan.Right_Left_3(
            ultrasonics["FrLH"].dis, ultrasonics["Fr"].dis, ultrasonics["FrRH"].dis)
    # 過去の値を使ってスムーズに走る
    elif config.mode_plan == "Right_Left_3_Records":
        steer_pwm_duty, throttle_pwm_duty = plan.Right_Left_3_Records(
            ultrasonics["FrLH"].dis, ultrasonics["Fr"].dis, ultrasonics["FrRH"].dis)
    # 右手法で走る
    elif config.mode_plan == "RightHand":
        steer_pwm_duty, throttle_pwm_duty = plan.RightHand(
            ultrasonics["FrRH"].dis, ultrasonics["RrRH"].dis)
    # 左手法で走る
    elif config.mode_plan == "LeftHand":
        steer_pwm_duty, throttle_pwm_duty = plan.LeftHand(
            ultrasonics["FrLH"].dis, ultrasonics["RrLH"].dis)
    # 右手法にPID制御を使ってスムーズに走る
    elif config.mode_plan == "RightHand_PID":
        steer_pwm_duty, throttle_pwm_duty = plan.RightHand_PID(
            ultrasonics["FrRH"], ultrasonics["RrRH"])
    # 左手法にPID制御を使ってスムーズに走る
    elif config.mode_plan == "LeftHand_PID":
        steer_pwm_duty, throttle_pwm_duty = plan.LeftHand_PID(
            ultrasonics["FrLH"], ultrasonics["RrLH"])
    # ニューラルネットを使ってスムーズに走る
    elif config.mode_plan == "NN":
        # 超音波センサ入力が変更できるように引数をリストにして渡す形に変更
        args = [ultrasonics[key].dis for key in config.ultrasonics_list]
        steer_pwm_duty, throttle_pwm_duty = plan.NN(model, *args)
    else:
        print("デフォルトの判断モードの選択ではありません, コードを書き換えてオリジナルのモードを実装しよう!")

    return steer_pwm_duty, throttle_pwm_duty


def control_joystick(joystick, motor, steer_pwm_duty, throttle_pwm_duty):
    """ジョイスティックで操作する場合は操舵値を上書き"""
    joystick.poll()
    mode = joystick.mode[0]
    if mode == "user":
        steer_pwm_duty = int(joystick.steer*config.JOYSTICK_STEERING_SCALE*100)
        throttle_pwm_duty = int(
            joystick.accel*config.JOYSTICK_THROTTLE_SCALE*100)
        if joystick.accel2:
            throttle_pwm_duty = int(config.FORWARD_S)
        elif joystick.accel1:
            throttle_pwm_duty = int(config.FORWARD_C)
    elif mode == "auto_str":
        throttle_pwm_duty = int(
            joystick.accel*config.JOYSTICK_THROTTLE_SCALE*100)
        if joystick.accel2:
            throttle_pwm_duty = int(config.FORWARD_S)
        elif joystick.accel1:
            throttle_pwm_duty = int(config.FORWARD_C)
    if joystick.recording:
        recording = True
    else:
        recording = False
    # コントローラでブレーキ
    if joystick.breaking:
        motor.breaking()
    return steer_pwm_duty, throttle_pwm_duty, recording


def planning_detection(steer_pwm_duty, throttle_pwm_duty):
    """認識結果をもとに走行判断"""
    global parking_mode
    objects = []
    detection_dict = {}
    message = ""
    norm_max = 1000  # ±1mの範囲で正規化
    norm_min = -1000

    if (len(detections)) >= 1:
        cnt_id = 0
        for detection in detections:
            object = Object
            object.name = labels[detection.label]
            object.pos[0] = detection.spatialCoordinates.x
            object.pos[1] = detection.spatialCoordinates.y
            object.pos[2] = detection.spatialCoordinates.z
            # リミット以上の検出物を除外しそれ以外をリスト格納する
            if detection.spatialCoordinates.z < DETECTION_DISTANCE_LIMIT:
                objects.append(object)
                detection_dict[object.name] = cnt_id
            cnt_id += 1

        # 右矢印を検出した場合、操舵を右に切る
        if "Right-arrow" in detection_dict and detections[detection_dict["Right-arrow"]].spatialCoordinates.x < 100:
            # offset_x = detections[detection_dict["Right-arrow"]
            #                       ].spatialCoordinates.x + OFFSET_ARROW_X
            # angle = np.rad2deg(
            #     np.arctan(offset_x/detections[detection_dict["Right-arrow"]
            #                                   ].spatialCoordinates.z))
            # converted_angle = (angle/90)*100  # 角度を-100から100の範囲に変換
            # steer_pwm_duty = converted_angle
            steer_pwm_duty = -100
            message = "右矢印を検出し操舵を右に切る"

        # 左矢印を検出した場合、操舵を左に切る
        elif "Left-arrow" in detection_dict and detections[detection_dict["Left-arrow"]].spatialCoordinates.x > -100:
            # offset_x = detections[detection_dict["Left-arrow"]
            #                       ].spatialCoordinates.x - OFFSET_ARROW_X
            # angle = np.rad2deg(
            #     np.arctan(offset_x/detections[detection_dict["Left-arrow"]
            #                                   ].spatialCoordinates.z))
            # converted_angle = (angle/90)*100
            # steer_pwm_duty = converted_angle
            steer_pwm_duty = 100
            message = "左矢印を検出し操舵を左に切る"

        # 青コーンのみ検出かつXが+側の場合、操舵を右に切る
        elif "Blue-cone" in detection_dict and detections[detection_dict["Blue-cone"]].spatialCoordinates.x > -100:
            steer_pwm_duty = -100
            message = "青コーンのみ検出かつXが+側の場合、操舵を右に切る"

        # 青コーンと緑コーンを検出した場合、中間に舵を切る
        elif "Blue-cone" in detection_dict and "Green-cone" in detection_dict:
            blue_x = detections[detection_dict["Blue-cone"]
                                ].spatialCoordinates.x
            green_x = detections[detection_dict["Green-cone"]
                                 ].spatialCoordinates.x
            target_x = (blue_x+green_x)/2
            # 正規化し100倍する
            steer_pwm_duty = (target_x-norm_min)/(norm_max-norm_min)*(-100)
            message = "青コーンと緑コーンを検出した場合、中間に舵を切る"

        # 緑コーンのみ検出かつXが-側の場合、操舵を左に切る
        elif "Green-cone" in detection_dict and detections[detection_dict["Green-cone"]].spatialCoordinates.x < 100:
            steer_pwm_duty = 100
            message = "緑コーンのみ検出かつXが-側の場合、操舵を左に切る"

        # 緑コーンと橙コーンを検出した場合、中間に舵を切る
        elif "Green-cone" in detection_dict and "Orange-cone" in detection_dict:
            green_x = detections[detection_dict["Green-cone"]
                                 ].spatialCoordinates.x
            orange_x = detections[detection_dict["Orange-cone"]
                                  ].spatialCoordinates.x
            target_x = (green_x+orange_x)/2
            steer_pwm_duty = (target_x-norm_min)/(norm_max-norm_min)*(-100)
            message = "緑コーンと橙コーンを検出した場合、中間に舵を切る"

        # 橙コーンのみ検出かつXが+側の場合、操舵を右に切る
        elif "Orange-cone" in detection_dict and detections[detection_dict["Orange-cone"]].spatialCoordinates.x > -100:
            steer_pwm_duty = -100
            message = "橙コーンのみ検出かつXが+側の場合、操舵を右に切る"

        # ピンクラインを検出したら減速する
        elif "Pink-line" in detection_dict and detections[detection_dict["Pink-line"]].spatialCoordinates.z < 200:
            throttle_pwm_duty = 80
            message = "ピンクラインを検出したら減速する"

        # 芝生を検出したら加速する
        elif "Shibafu" in detection_dict and detections[detection_dict["Shibafu"]].spatialCoordinates.z < 600:
            throttle_pwm_duty = 120
            message = "芝生を検出したら加速する"

        # パーキングモード時、P1-Greenに向かう
        elif parking_mode == True and "P1-Green" in detection_dict:
            # x軸が100mm以上離れていたら操舵補正する
            if detections[detection_dict["P1-Green"]].spatialCoordinates.x > 100:
                steer_pwm_duty = -80
            elif detections[detection_dict["P1-Green"]].spatialCoordinates.x < -100:
                steer_pwm_duty = 80
            if detections[detection_dict["P1-Green"]].spatialCoordinates.z < 200:
                throttle_pwm_duty = 50
            elif detections[detection_dict["P1-Green"]].spatialCoordinates.z < 100:
                throttle_pwm_duty = 0

        print("*******************************************************************")
        print()
        print("検出物：", detection_dict)
        print(message)

        # 上限値超えを修正
        if steer_pwm_duty > 100:
            steer_pwm_duty = 100
        elif steer_pwm_duty < -100:
            steer_pwm_duty = -100

    return steer_pwm_duty, throttle_pwm_duty


def detect() -> None:
    """画像認識スレッド"""
    global detections
    global labels

    # 画像認識の初期化
    oakd_spatial_yolo = OakdSpatialYolo(
        config_path=args.config,
        model_path=args.model,
        fps=args.fps,
        save_fps=args.save_fps,
    )
    labels = oakd_spatial_yolo.get_labels()
    print('******************************************************')
    print('*************** Enterを押して走行開始! ***************')
    print('******************************************************')
    while True:
        frame = None
        try:
            frame, detections = oakd_spatial_yolo.get_frame()
        except BaseException:
            print("===================")
            print("get_frame() error! Reboot OAK-D.")
            print("If reboot occur frequently, Bandwidth may be too much.")
            print("Please lower FPS.")
            print("===================")
        if frame is not None:
            oakd_spatial_yolo.display_frame("nn", frame, detections)
        if cv2.waitKey(1) == ord("q"):
            end = True
            break
    oakd_spatial_yolo.close()


def run() -> None:
    """走行用スレッド"""
    global start_time
    global current_time
    global parking_mode

    # データ記録用配列作成
    d = np.zeros(config.N_ultrasonics)
    d_stack = np.zeros(config.N_ultrasonics+3)
    recording = True

    # 操舵、駆動モーターの初期化
    motor = Motor()
    motor.set_throttle_pwm_duty(config.STOP)
    motor.set_steer_pwm_duty(config.NUTRAL)

    # 超音波センサの初期化
    # 別々にインスタンス化する例　ultrasonic_RrLH = ultrasonic.Ultrasonic("RrLH")
    # 一気にnameに"RrLH"等をultrasonics_listから入れてインスタンス化
    ultrasonics = {name: ultrasonic.Ultrasonic(
        name=name) for name in config.ultrasonics_list}
    print(" 下記の超音波センサを利用")
    print(" ", config.ultrasonics_list)

    # 操作判断プランナーの初期化
    plan = planner.Planner(config.mode_plan)

    # NNモデルの読み込み
    model = None
    if config.HAVE_NN:
        # NNモデルの初期化
        # 使う超音波センサの数、出力数、隠れ層の次元、隠れ層の数
        model = train_pytorch.NeuralNetwork(
            len(config.ultrasonics_list), 2,
            config.hidden_dim, config.num_hidden_layers)
        # 保存したモデルをロード
        print("\n保存したモデルをロードします: ", config.model_path)
        train_pytorch.load_model(
            model, config.model_path, None, config.model_dir)
        print(model)

    # コントローラーの初期化
    if config.HAVE_CONTROLLER:
        joystick = Joystick()
        if joystick.HAVE_CONTROLLER == False:
            config.HAVE_CONTROLLER = False
        mode = joystick.mode[0]
        print("Starting mode: ", mode)

    # 一時停止（Enterを押すとプログラム実行開始）
    # print('*************** Enterを押して走行開始! ***************')
    input()

    # 途中でモータースイッチを切り替えたとき用に再度モーター初期化
    # 初期化に成功するとピッピッピ！と３回音がなる、失敗時（PWMの値で約370-390以外の値が入りっぱなし）はピ...ピ...∞
    motor.set_throttle_pwm_duty(config.STOP)

    # 開始時間
    start_time = time.time()

    # ここから走行ループ
    try:
        while True:
            # 現在時間を格納
            current_time = time.time()

            # 認知（超音波センサ計測）
            message = measure_ultrasonic(d, ultrasonics)

            # 判断（プランニング）
            steer_pwm_duty, throttle_pwm_duty = planning_ultrasonic(
                plan, ultrasonics, model)
            steer_pwm_duty = steer_pwm_duty * 1.3

            # 画像認識
            steer_pwm_duty, throttle_pwm_duty = planning_detection(
                steer_pwm_duty, throttle_pwm_duty)

            # 操作（ステアリング、アクセル）
            if config.HAVE_CONTROLLER:
                steer_pwm_duty, throttle_pwm_duty, recording = control_joystick(
                    joystick, motor, steer_pwm_duty, throttle_pwm_duty)

            # 一定時間経過後駐車モードにする
            if current_time > start_time + PARKING_TIME:
                parking_mode = True

            # モータードライバーに出力をセット
            motor.set_steer_pwm_duty(steer_pwm_duty)
            motor.set_throttle_pwm_duty(throttle_pwm_duty)

            # 記録（タイムスタンプと距離データを配列に記録）
            ts = time.time()
            ts_run = round(ts-start_time, 2)
            if recording:
                d_stack = np.vstack(
                    (d_stack, np.insert(d, 0, [ts, steer_pwm_duty, throttle_pwm_duty]),))

            # 全体の状態を出力
            if mode == 'auto':
                mode = config.mode_plan
            if config.plotter:
                print(message)
            else:
                print(
                    "*******************************************************************")
                print("Rec:{0}, Mode:{1}, RunTime:{2:>5}".format(
                    recording, mode, ts_run))
                print("Str:{0:>1}, Thr:{1:>1}".format(
                    int(steer_pwm_duty), int(throttle_pwm_duty)))
                print("Uls:[ {0}]".format(message))

            # 後退/停止操作（簡便のため、判断も同時に実施）
            if config.mode_recovery == "None":
                pass
            elif config.mode_recovery == "Back" and mode != "user":
                # 後退
                plan.Back(ultrasonics["Fr"],
                          ultrasonics["FrRH"], ultrasonics["FrLH"])
                if plan.flag_back == True:
                    for _ in range(config.recovery_braking):
                        # motor.set_steer_pwm_duty(config.NUTRAL)
                        motor.set_steer_pwm_duty(-70)  # バック時ハンドルを右に切る
                        motor.set_throttle_pwm_duty(config.REVERSE)
                        time.sleep(config.recovery_time)
                else:
                    pass
            elif config.mode_recovery == "Stop" and mode != "user":
                # 停止
                plan.Stop(ultrasonics["Fr"])
                if plan.flag_stop == True:
                    motor.set_steer_pwm_duty(config.recovery_str)
                    for _ in range(config.recovery_braking):
                        motor.set_throttle_pwm_duty(config.STOP)
                        time.sleep(0.02)
                        motor.set_throttle_pwm_duty(config.REVERSE)
                        time.sleep(0.1)
                        time.sleep(config.recovery_time /
                                   config.recovery_braking)
                    motor.set_throttle_pwm_duty(config.STOP)
                    plan.flag_stop = False
                    print("一時停止、Enterを押して走行再開!")
                    input()

    finally:
        # 終了処理
        print('\n停止')
        motor.set_throttle_pwm_duty(config.STOP)
        motor.set_steer_pwm_duty(config.NUTRAL)
        GPIO.cleanup()
        header = "Tstamp,Str,Thr,"
        for name in config.ultrasonics_list:
            header += name + ","
        header = header[:-1]
        np.savetxt(config.record_filename,
                   d_stack[1:], delimiter=',',  fmt='%10.2f', header=header, comments="")
        print('記録停止')
        print("記録保存--> ", config.record_filename)
        if config.HAVE_CAMERA:
            print("画像保存--> ", config.image_dir)
        sys.exit()


def main():
    t1 = threading.Thread(target=run)  # 走行用スレッド
    t2 = threading.Thread(target=detect)  # 画像認識用スレッド
    t1.start()
    t2.start()
    t1.join()
    t2.join()


if __name__ == "__main__":
    main()
