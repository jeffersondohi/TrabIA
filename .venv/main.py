import os
import cv2
import sys
import tkinter as tk
from tkinter import filedialog

#teste
def main(source=0):
    capture = cv2.VideoCapture(source) #inicia captura de video
    if not capture.isOpened():
        print(f"Erro ao abrir a fonte de vídeo: {source}")
        return

    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    in_width = 300
    in_height = 300
    mean = [104, 117, 123]
    conf_threshold = 0.7

    while cv2.waitKey(1) != 27:
        has_frame, frame = capture.read()
        if not has_frame:
            break
        frame = cv2.flip(frame, 1)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
                y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
                x_right_top = int(detections[0, 0, i, 5] * frame_width)
                y_right_top = int(detections[0, 0, i, 6] * frame_height)

                cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
                label = "Confidence: %.4f" % confidence
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv2.rectangle(
                    frame,
                    (x_left_bottom, y_left_bottom - label_size[1]),
                    (x_left_bottom + label_size[0], y_left_bottom + base_line),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        t, _ = net.getPerfProfile()
        label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow(win_name, frame)

    capture.release()
    cv2.destroyWindow(win_name)
    choose_source()


def choose_source():
    def use_camera():
        root.destroy()
        main(0)

    def use_video():
        root.destroy()
        video_path = filedialog.askopenfilename(title="Selecione o arquivo de video",
                                                filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
        if video_path:
            main(video_path)

    def exit_program():
        root.destroy()
        sys.exit()

    root = tk.Tk()
    root.title("Choose Source")

    tk.Label(root, text="Deseja fazer reconhecimento facial pela camera ou video?", pady=20).pack()

    tk.Button(root, text="Usar camera", command=use_camera, width=20, pady=10).pack()
    tk.Button(root, text="Usar arquivo de video", command=use_video, width=20, pady=10).pack()
    tk.Button(root, text="Sair", command=exit_program, width=20, pady=10).pack()

    root.mainloop()


if __name__ == "__main__":
    choose_source()
