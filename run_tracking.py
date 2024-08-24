#!/usr/bin/env python3
# coding: utf-8

# 一般的な外部ライブラリ
import sys
import json
import time
import numpy as np
import multiprocessing
from multiprocessing import Process
import argparse
import RPi.GPIO as GPIO

# togikaidriveのモジュール
import ultrasonic
from motor import Motor
import planner
import config

# from lib.oakd_yolo import OakdYolo
from lib.oakd_tracking_yolo import OakdTrackingYolo

GPIO.setwarnings(False)
# GPIOピン番号の指示方法
GPIO.setmode(GPIO.BOARD)
GPIO.setup(config.e_list, GPIO.IN)
GPIO.setup(config.t_list, GPIO.OUT, initial=GPIO.LOW)

# 以下はconfig.pyでの設定によりimport
if config.HAVE_CONTROLLER:
    from joystick import Joystick
if config.HAVE_CAMERA:
    import camera_multiprocess
if config.HAVE_IMU:
    import gyro
if config.HAVE_NN:
    import train_pytorch


def measure_ultrasonic(d, ultrasonics):
    # 認知（超音波センサ計測）
    # RrRHセンサ距離計測例：dis_RrRH = ultrasonic_RrRH.()
    # 下記では一気に取得
    message = ""
    for i, name in enumerate(config.ultrasonics_list):
        d[i] = ultrasonics[name].measure()
        # message += name + ":" + str(round(ultrasonics[name].dis,2)).rjust(7, ' ') #Thony表示用にprint変更
        message += name + ":" + \
            "{:>4}".format(round(ultrasonics[name].dis)) + ", "
        # サンプリングレートを調整する場合は下記をコメントアウト外す
        # time.sleep(sampling_cycle)
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
    # 評価中
    elif config.mode_plan == "NN":
        # 超音波センサ入力が変更できるように引数をリストにして渡す形に変更
        args = [ultrasonics[key].dis for key in config.ultrasonics_list]
        steer_pwm_duty, throttle_pwm_duty = plan.NN(model, *args)
        # steer_pwm_duty, throttle_pwm_duty  = plan.NN(model, ultrasonics["FrLH"].dis, ultrasonics["Fr"].dis, ultrasonics["FrRH"].dis)
    else:
        print("デフォルトの判断モードの選択ではありません, コードを書き換えてオリジナルのモードを実装しよう!")
        # break
    return steer_pwm_duty, throttle_pwm_duty


def control_joystick(joystick, motor):
    # ジョイスティックで操作する場合は上書き
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


def object_tracking(oakd_tracking_yolo):
    frame = None
    tracklets = []
    try:
        frame, detections, tracklets = oakd_tracking_yolo.get_frame()
    except BaseException:
        print("===================")
        print("get_frame() error! Reboot OAK-D.")
        print("If reboot occur frequently, Bandwidth may be too much.")
        print("Please lower FPS.")
        print("===================")
        # break

    if tracklets is not None:
        if (len(tracklets)) > 0:
            for tracklet in tracklets:
                print(labels[tracklet.label])
                if labels[tracklet.label] == "right-arrow":
                    steer_pwm_duty = 0
                elif labels[tracklet.label] == "left-arrow":
                    steer_pwm_duty = 0

    return steer_pwm_duty
    # if frame is not None:
    #     oakd_tracking_yolo.display_frame("nn", frame, tracklets)
    # if cv2.waitKey(1) == ord("q"):
    #     end = True
    #     break


def main(args) -> None:
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

    # 　imuの初期化
    if config.HAVE_IMU:
        imu = gyro.BNO055()
        # 計測例
        # angle, acc, gyr = imu.measure_set()

    # コントローラーの初期化
    if config.HAVE_CONTROLLER:
        joystick = Joystick()
        if joystick.HAVE_CONTROLLER == False:
            config.HAVE_CONTROLLER = False
        mode = joystick.mode[0]
        print("Starting mode: ", mode)

    bird_frame = True
    orbit = True
    spatial_frame = False
    if args.disable_orbit:
        orbit = False
    # spatial_frameを有効化した場合、bird_frameは無効化
    if args.spatial_frame:
        bird_frame = False
        spatial_frame = True
        orbit = False
    end = False

    # 画像認識の初期化
    # oakd_yolo = OakdYolo(args.config, args.model, args.fps, save_fps=args.save_fps)
    oakd_tracking_yolo = OakdTrackingYolo(
        config_path=args.config,
        model_path=args.model,
        fps=args.fps,
        cam_debug=args.display_camera,
        robot_coordinate=args.robot_coordinate,
        # show_bird_frame=bird_frame,
        # show_spatial_frame=spatial_frame,
        show_orbit=orbit,
        log_path=args.log_path,
    )
    oakd_tracking_yolo.update_bird_frame_distance(10000)
    # labels = oakd_yolo.get_labels()

    # 一時停止（Enterを押すとプログラム実行開始）
    print('*************** Enterを押して走行開始! ***************')
    input()

    # 途中でモータースイッチを切り替えたとき用に再度モーター初期化
    # 初期化に成功するとピッピッピ！と３回音がなる、失敗時（PWMの値で約370-390以外の値が入りっぱなし）はピ...ピ...∞
    motor.set_throttle_pwm_duty(config.STOP)

    # fpv
    # pass

    # 開始時間
    start_time = time.time()

    # ここから走行ループ
    try:
        while True:

            # 認知（超音波センサ計測）
            message = measure_ultrasonic(d, ultrasonics)
            # 判断（プランニング）
            steer_pwm_duty, throttle_pwm_duty = planning_ultrasonic(
                plan, ultrasonics, model)

            # 画像認識
            # object_detection(oakd_yolo, labels)
            steer_pwm_duty = object_tracking(oakd_tracking_yolo)

            # 操作（ステアリング、アクセル）
            if config.HAVE_CONTROLLER:
                steer_pwm_duty, throttle_pwm_duty, recording = control_joystick(
                    joystick, motor)

            # モータードライバーに出力をセット
            # 補正（動的制御）
            # Gthr:スロットル（前後方向）のゲイン、Gstr:ステアリング（横方向）のゲイン
            # ヨー角の角速度でオーバーステア/スリップに対しカウンターステア
            if config.mode_plan == "GCounter":
                imu.GCounter()
                motor.set_steer_pwm_duty(steer_pwm_duty * (1 - 2 * imu.Gstr))
                motor.set_throttle_pwm_duty(
                    throttle_pwm_duty * (1 - 2 * imu.Gthr))
            # ヨー角の角速度でスロットル調整
            # 未実装
            # elif config.mode_plan == "GVectoring":
            #    imu.GVectoring()
            else:
                motor.set_steer_pwm_duty(steer_pwm_duty)
                motor.set_throttle_pwm_duty(throttle_pwm_duty)

            # 記録（タイムスタンプと距離データを配列に記録）
            ts = time.time()
            ts_run = round(ts-start_time, 2)
            if recording:
                d_stack = np.vstack(
                    (d_stack, np.insert(d, 0, [ts, steer_pwm_duty, throttle_pwm_duty]),))
                # 画像保存 ret:カメラ認識、img：画像
                # if config.HAVE_CAMERA and not config.fpv:
                #     ret, img = cam.read()
                #     cam.save(img, ts, steer_pwm_duty, throttle_pwm_duty, config.image_dir)

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
                    steer_pwm_duty, throttle_pwm_duty))
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
                        motor.set_steer_pwm_duty(config.NUTRAL)
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
                    # break

    finally:
        # 終了処理
        print('\n停止')
        # oakd_yolo.close()
        oakd_tracking_yolo.close()
        motor.set_throttle_pwm_duty(config.STOP)
        motor.set_steer_pwm_duty(config.NUTRAL)
        GPIO.cleanup()
        header = "Tstamp,Str,Thr,"
        for name in config.ultrasonics_list:
            header += name + ","
        header = header[:-1]
        np.savetxt(config.record_filename,
                   d_stack[1:], delimiter=',',  fmt='%10.2f', header=header, comments="")
        # np.savetxt(config.record_filename, d_stack[1:], fmt='4f',header=header, comments="")
        print('記録停止')
        print("記録保存--> ", config.record_filename)
        if config.HAVE_CAMERA:
            print("画像保存--> ", config.image_dir)
        sys.exit()

    # header ="Tstamp, Str, Thr, "
    # for name in config.ultrasonics_list:
    #     header += name + ", "
    # header = header[:-1]


if __name__ == "__main__":
    # 引数設定
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Provide model name or model path for inference",
        default="yolov7tiny_coco_416x416",
        # default="models/minicar_20240815.blob",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Provide config path for inference",
        default="json/yolov7tiny_coco_416x416.json",
        # default="json/minicar_20240815.json",
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
        "-d",
        "--display_camera",
        help="Display camera rgb and depth frame",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--robot_coordinate",
        help="Convert object pos from camera coordinate to robot coordinate",
        action="store_true",
    )
    parser.add_argument(
        "--spatial_frame",
        help="Display spatial frame instead of bird frame",
        action="store_true",
    )
    parser.add_argument(
        "--disable_orbit",
        help="Disable display tracked orbit on bird frame",
        action="store_true",
    )
    parser.add_argument(
        "--log_path",
        help="Path to save orbit data",
        type=str,
    )
    # parser.add_argument(
    #     "-s",
    #     "--save_fps",
    #     help="Image save fps. If it's > 0, images and video will be saved.",
    #     default=0,
    #     type=int,
    # )
    args = parser.parse_args()

    json_open = open(args.config, 'r')
    json_load = json.load(json_open)
    labels = json_load['mappings']['labels']

    main(args)
