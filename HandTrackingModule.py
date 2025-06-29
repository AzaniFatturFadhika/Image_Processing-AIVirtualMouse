"""
Hand Tracking Module (Modul Pelacakan Tangan)
Oleh: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone/

Dimodifikasi untuk menambahkan indikator tangan Kiri/Kanan.
"""

import cv2  # Library untuk operasi pada gambar dan video
import mediapipe as mp  # Library untuk deteksi dan pengolahan tangan
import time  # Library untuk mengakses waktu
import math  # Library untuk operasi matematika
import numpy as np  # Library untuk operasi array multidimensi


# Kelas HandDetector membungkus semua proses deteksi tangan.
class HandDetector():
    # Metode inisialisasi saat objek HandDetector dibuat.
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        """
        :param mode: Jika True, mode gambar statis. Jika False, mode video (lebih baik untuk tracking).
        :param maxHands: Jumlah maksimal tangan yang akan dideteksi.
        :param detectionCon: Ambang batas kepercayaan deteksi (misal: 0.5 = 50%).
        :param trackCon: Ambang batas kepercayaan pelacakan (misal: 0.5 = 50%).
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Menginisialisasi solusi 'hands' dari MediaPipe.
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, # Mode gambar statis atau video
            max_num_hands=self.maxHands, # Jumlah tangan maks
            min_detection_confidence=self.detectionCon, # Kepercayaan deteksi
            min_tracking_confidence=self.trackCon # Kepercayaan pelacakan
        )
        # Utilitas untuk menggambar landmark dan koneksi tangan.
        self.mpDraw = mp.solutions.drawing_utils
        # ID landmark untuk ujung setiap jari (jempol, telunjuk, tengah, manis, kelingking).
        self.tipIds = [4, 8, 12, 16, 20]
        # Daftar untuk menyimpan jenis tangan yang terdeteksi ('Left' atau 'Right')
        self.handedness = []

    def findHands(self, img, draw=True):
        # MediaPipe bekerja dengan gambar RGB, sedangkan OpenCV menggunakan BGR. Jadi, kita konversi.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Memproses gambar untuk menemukan tangan. Hasilnya disimpan di self.results.
        self.results = self.hands.process(imgRGB)

        # Mengosongkan dan mengisi kembali daftar handedness setiap frame
        self.handedness = []
        if self.results.multi_handedness:
            for hand_handedness in self.results.multi_handedness:
                self.handedness.append(hand_handedness.classification[0].label)

        # Jika landmark tangan terdeteksi (multi_hand_landmarks tidak kosong).
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: # Jika draw=True, gambar kerangka tangan pada gambar asli.
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        # Inisialisasi daftar untuk menyimpan koordinat x, y, dan bounding box.
        xList = []
        yList = []
        bbox = []
        # Daftar ini akan berisi [id_landmark, x, y] untuk setiap titik.
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] # Pilih tangan berdasarkan indeks (default tangan pertama).
            for id, lm in enumerate(myHand.landmark):
                # Mendapatkan dimensi gambar untuk mengonversi koordinat normalisasi ke piksel.
                h, w, c = img.shape
                # Koordinat landmark (lm.x, lm.y) adalah rasio (0-1). Ubah ke koordinat piksel.
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw: # Jika draw=True, gambar lingkaran pada setiap landmark.
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Menentukan kotak pembatas (bounding box) di sekitar tangan.
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw: # Jika draw=True, gambar kotak pembatas di sekitar tangan.
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self, handNo=0):
        # Fungsi untuk mendeteksi jari mana yang terangkat.
        fingers = []
        # --- Logika untuk Jempol (Dinamis berdasarkan Tangan Kiri/Kanan) ---
        # Logika ini bekerja pada gambar yang sudah di-flip (efek cermin)
        if len(self.lmList) != 0 and len(self.handedness) > handNo:
            hand_type = self.handedness[handNo]
            # Untuk tangan kanan (di cermin), jempol di kiri. x_tip < x_joint.
            if hand_type == "Right":
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # Untuk tangan kiri (di cermin), jempol di kanan. x_tip > x_joint.
            else:  # Left Hand
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        elif len(self.lmList) != 0: # Fallback jika handedness tidak terdeteksi (jarang terjadi)
            # Gunakan logika tangan kanan sebagai default
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Logika untuk 4 Jari Lainnya (Telunjuk hingga Kelingking):
        # Memeriksa apakah ujung jari (y) berada di atas sendi dua tingkat di bawahnya.
        # Di OpenCV, koordinat y yang lebih kecil berarti posisi lebih atas.
        if len(self.lmList) != 0:
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        # Mendapatkan koordinat piksel untuk dua titik landmark (p1 dan p2).
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        # Menghitung titik tengah antara p1 dan p2.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:  # Jika draw=True, gambar garis dan lingkaran untuk visualisasi.
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        # Menghitung jarak Euclidean antara dua titik.
        length = math.hypot(x2 - x1, y2 - y1)

        # Mengembalikan jarak, gambar yang sudah dimodifikasi, dan info koordinat.
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # Fungsi utama untuk pengujian mandiri modul ini.
    # Kode di sini hanya akan berjalan jika file ini dieksekusi secara langsung.
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Menggunakan webcam utama.
    detector = HandDetector(maxHands=1) # Batasi deteksi hanya untuk satu tangan
    while True:
        success, img = cap.read()
        # Flip gambar secara horizontal agar seperti cermin
        img = cv2.flip(img, 1)
        # Temukan tangan dan gambar kerangkanya
        img = detector.findHands(img)
        # Dapatkan posisi landmark dan kotak pembatas
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:  # Jika tangan terdeteksi...
            # Periksa apakah informasi tangan (kiri/kanan) tersedia
            if detector.handedness:
                # Ambil jenis tangan (misal: 'Left' atau 'Right')
                hand_type = detector.handedness[0]
                
                # Dapatkan dimensi lebar dan tinggi dari frame video
                h, w, c = img.shape
                
                # Definisikan properti kotak indikator
                box_size = 150
                margin = 20
                
                # Jika tangan kanan terdeteksi
                if hand_type == "Right":
                    # Gambar kotak merah di sisi kanan atas
                    cv2.rectangle(img, (w - margin - box_size, margin), (w - margin, margin + box_size), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Kanan", (w - margin - box_size + 25, margin + box_size - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                # Jika tangan kiri terdeteksi
                elif hand_type == "Left":
                    # Gambar kotak merah di sisi kiri atas
                    cv2.rectangle(img, (margin, margin), (margin + box_size, margin + box_size), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Kiri", (margin + 40, margin + box_size - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # Cetak jari yang terangkat dan jenis tangan ke konsol
            fingers = detector.fingersUp()
            print(fingers, detector.handedness)


        # Menghitung dan menampilkan FPS.
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, h - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

        # Menampilkan gambar.
        cv2.imshow("Image", img)
        # Hentikan loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Melepaskan webcam dan menutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()


# Blok ini memastikan bahwa fungsi main() hanya dipanggil saat skrip dijalankan langsung.
if __name__ == "__main__":
    main()