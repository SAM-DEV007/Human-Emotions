import tensorflow as tf
import mediapipe as mp
import numpy as np

import os
import cv2
import shutil
import sys


def write_word(frame, word: str, x: int, y: int, size: float, color: tuple = (0, 0, 0)):
    '''Writes word in the frame'''

    cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA, False)


def create_cache(cache_path):
    '''Creates Cache folder'''

    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        return
    shutil.rmtree(cache_path)
    os.mkdir(cache_path)


if __name__ == '__main__':
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model\\Model_Data\\Model.h5'
    model = tf.keras.models.load_model(model_path)

    cache_path = model_path = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Cache\\'

    vid = cv2.VideoCapture(0)

    _, frame = vid.read()
    h, w, _ = frame.shape

    face_detection = mp.solutions.face_detection.FaceDetection()

    # Emotions
    emo = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

    create_cache(cache_path)
    sys.stdout = open(os.devnull, 'w') # Disables print

    while True:
        _, frame = vid.read()
        frame = cv2.flip(frame, 1)

        face_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = face_detection.process(face_frame)

        if face.detections:
            for detection in face.detections:
                face_data = detection.location_data
                box_data = face_data.relative_bounding_box
                x = round(box_data.xmin * w)
                y = round(box_data.ymin * h)
                wi = round(box_data.width * w)
                he = round(box_data.height * h)
            
                if 0 < x < w and 0 < y < h and 0 < x+wi < w and 0 < y+he < h:
                    gray_frame = frame[y: y+he, x: x+wi]
                    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(cache_path+'img.png', gray_frame)

                    img = tf.keras.utils.image_dataset_from_directory(cache_path, labels=None, image_size=(48, 48))

                    prediction = np.round(np.squeeze(model.predict(img)), 4)
                    predict = np.argmax(prediction)
                    stat = {emo[i]: val for i, val in enumerate(prediction)}

                    y_val = 10

                    create_cache(cache_path)
                    cv2.rectangle(frame, (x, y), (x+wi, y+he), (0, 255, 0), 2)

                    write_word(frame, emo[predict], x, y-10, 0.75)
                    for txt in stat.keys():
                        write_word(frame, f'{txt}. Prediction - {str(stat[txt])[:4]}', x+wi+10, y+y_val, 0.6, (255, 255, 255))
                        y_val += 25
    
        # Shows and records the video
        cv2.imshow('Cam', frame)

        # Closes the window
        # Q button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Esc button
        if cv2.waitKey(1) == 27:
            break

        # X button on the top of the window
        if cv2.getWindowProperty('Cam', cv2.WND_PROP_VISIBLE) < 1:
            break

    vid.release()
    cv2.destroyAllWindows()
