import csv
import cv2
import openpyxl
import os
import mediapipe as mp
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


def calcola_media_colonne(file_excel):
    wb = openpyxl.load_workbook(file_excel)
    foglio = wb.worksheets[0]

    media_colonne = []
    etichette = []
    nomi_video = []

    for riga in foglio.iter_rows(min_row=2, max_row=2, min_col=2):
        for cella in riga:
            if isinstance(cella.value, str):
                if cella.value.startswith("The person in the video appears"):
                    etichetta = "-" + cella.value.split("-")[-1].strip()
                else:
                    etichetta = cella.value.strip()
                etichette.append(etichetta)
            else:
                etichette.append('')

    for cella in foglio[1][1:]:
        if isinstance(cella.value, str):
            nomi_video.append(cella.value)
        else:
            nomi_video.append('')

    for colonna in foglio.iter_cols(min_row=3, min_col=2):
        valori_colonna = [cella.value for cella in colonna if isinstance(cella.value, (int, float))]
        if valori_colonna:
            media = sum(valori_colonna) / len(valori_colonna)
        else:
            media = 0
        media_colonne.append(media)

    risultato = []
    ultimo_nome_video = ''

    for i in range(len(media_colonne)):
        sub_array = media_colonne[i:i + 4]
        if sub_array:
            indice_massimo = sub_array.index(max(sub_array))
            massimo = max(sub_array)
            etichetta = etichette[i + indice_massimo] if i + indice_massimo < len(etichette) else ''
            nome_video = nomi_video[i] if i < len(nomi_video) else ''
            if nome_video != ultimo_nome_video:
                risultato.append((nome_video, etichetta))
                ultimo_nome_video = nome_video
    return risultato


file_excel = 'datasets_xlsx/responses_test.xlsx'
risultato = calcola_media_colonne(file_excel)
file_csv = './dativideoTest.csv'

with open(file_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Nome Video', 'Emozione'])
    for nome_video, etichetta in risultato:
        if nome_video.startswith("VID_RGB_000"):
            nome_video = nome_video.replace("VID_RGB_000", "").strip()
        writer.writerow([nome_video, etichetta])
        print("Nome video:", nome_video)
        print("Emozione:", etichetta)
        print()

print("Dati salvati correttamente nel file CSV:", file_csv)


def apply_landmarks(frame, mp_holistic, target_width, target_height):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), 0)
    frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))

    results = mp_holistic.process(frame_rgb)
    frame_with_landmarks = frame_rgb.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        frame_with_landmarks, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(
        frame_with_landmarks, results.face_landmarks)

    return frame_with_landmarks


def process_video_folder(input_folder, output_folder, target_width, target_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mp_holistic = mp.solutions.holistic.Holistic()
    processed_videos = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.avi', '.mp4', '.mkv', '.mov')):
            video_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]
            output_subfolder = os.path.join(output_folder, video_name)
            index = 1
            while os.path.exists(output_subfolder):
                output_subfolder = os.path.join(output_folder, f"{video_name}_{index}")
                index += 1
            os.makedirs(output_subfolder)

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_with_landmarks_list = []
            frame_count = 0
            start_time = 0
            end_time = start_time + 1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = frame_count / fps

                if current_time >= start_time and current_time < end_time:
                    frame_with_landmarks = apply_landmarks(frame, mp_holistic, target_width, target_height)
                    frame_with_landmarks_list.append(frame_with_landmarks)
                    output_path = os.path.join(output_subfolder, f"frame{frame_count}.jpg")
                    cv2.imwrite(output_path, frame_with_landmarks)

                frame_count += 1

                if current_time >= end_time:
                    break

            cap.release()

            processed_videos.append((video_name, frame_with_landmarks_list))

    mp_holistic.close()

    return processed_videos


# ...

def test_lstm(file_csv, nomi_cartelle):
    df = pd.read_csv(file_csv)
    emotions = {'-Happy': 0, '-Sad': 1, '-Angry': 2, '-Neutral': 3}

    X = []
    y = []

    # Iterazione sui nomi delle cartelle
    for folder_name in nomi_cartelle:
        video_name = " " + folder_name  # Nome della cartella video

        if video_name in df['Nome Video'].values:
            emotion = df.loc[df['Nome Video'] == video_name, 'Emozione'].values[0]
            emotion_label = emotions[emotion]

            frame_array = []  # Array per salvare i frame

            folder_path = os.path.abspath("outputframeTest\\" + folder_name)
            framesInVideoDir = os.listdir(folder_path)
            framesInVideoDirSorted = sorted(framesInVideoDir, key=lambda x: int(''.join(filter(str.isdigit, x))))

            for frame_file in framesInVideoDirSorted:
                frame_path = os.path.join(folder_path, frame_file)
                frame = cv2.imread(frame_path)
                frame = cv2.resize(frame, (90, 160))  # Riduci la dimensione dell'immagine
                frame_array.append(frame)

            X.append(frame_array)  # Aggiungi l'array dei frame con landmarks
            y.append(emotion_label)

    print(y)
    # Codifica delle etichette
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = np.array(X)
    y = np.array(y)

    # Load the pre-trained model
    model = load_model('emotion_lstm_model.h5')

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(X, y)
    print("Accuratezza sul set di test: {:.2f}%".format(test_accuracy * 100))

    # Make predictions on the test data
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print classification report
    print(classification_report(y, y_pred_classes, target_names=emotions.keys()))

    # Print confusion matrix
    print(confusion_matrix(y, y_pred_classes))


def get_folder_names(output_folder):
    if os.path.isdir(output_folder):
        nomi_cartelle = [nome for nome in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, nome))]
        return nomi_cartelle
    else:
        print('Il percorso specificato non corrisponde a una cartella.')
        return []


# Esempio di utilizzo della funzione test_lstm()
input_folder = './videosTest'
output_folder = './outputframeTest'
target_width = 180
target_height = 320
file_csv = './dativideoTest.csv'

processed_videos = process_video_folder(input_folder, output_folder, target_width, target_height)
nomi_cartelle = get_folder_names(output_folder)
test_lstm(file_csv, nomi_cartelle)
