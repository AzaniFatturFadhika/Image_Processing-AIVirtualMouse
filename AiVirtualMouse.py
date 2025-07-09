# Mengimpor library yang diperlukan
import cv2
import numpy as np
import HandTrackingModule as htm # Modul kustom untuk mendeteksi tangan
import time
import autopy  # Library untuk mengontrol mouse
import pyautogui  # Library untuk screenshot

##########################
# Pengaturan Awal
wCam, hCam = 640, 480  # Lebar dan tinggi jendela kamera
frameR = 100  # Frame Reduction: Mengurangi area aktif untuk kontrol mouse agar lebih stabil
smoothening = 7  # Faktor untuk memperhalus gerakan mouse
SCROLL_SENSITIVITY = 0.2 # Kontrol kecepatan scroll. Semakin KECIL, semakin SENSITIF.
#########################
 
# Variabel Waktu dan Posisi
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
last_scroll_y = None # Untuk melacak posisi Y terakhir saat scrolling
 
# Inisialisasi Webcam
cap = cv2.VideoCapture(0)  # Menggunakan webcam utama (indeks 0)
cap.set(3, wCam)  # Mengatur lebar frame
cap.set(4, hCam)  # Mengatur tinggi frame
 
# Inisialisasi Modul Deteksi Tangan
detector = htm.HandDetector(maxHands=1, detectionCon=0.75, trackCon=0.75)
 
# Mendapatkan ukuran layar monitor
wScr, hScr = autopy.screen.size()  # wScr: lebar layar, hScr: tinggi layar
 
# --- Manajemen Mode dan State ---
class Mode:
    IDLE = "IDLE"
    TRACKING = "TRACKING"
 
current_mode = Mode.IDLE
gesture_timer = None
GESTURE_HOLD_TIME_SECS = 1.5  # Waktu (detik) untuk menahan gestur untuk screenshot
 

while True:
    # 1. Menangkap frame dari webcam dan menemukan landmark tangan
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Membalik frame secara horizontal (efek cermin) agar gerakan intuitif
    img = detector.findHands(img)  # Mendeteksi tangan dan menggambar kerangkanya
    lmList, bbox = detector.findPosition(img)  # Mendapatkan daftar posisi landmark tangan

    # Menggambar area aktif
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    # 2. Jika tangan terdeteksi, proses gestur berdasarkan mode saat ini
    if len(lmList) != 0:
        fingers = detector.fingersUp(0) # Menggunakan handNo=0 karena maxHands=1

        # GESTUR GLOBAL: Kembali ke mode IDLE jika semua jari terangkat.
        # Berfungsi sebagai gestur "reset" atau "stop" universal.
        # [jempol, telunjuk, tengah, manis, kelingking]

        if fingers == [1, 1, 1, 1, 1] or fingers == [0,0,0,0,0]:
            current_mode = Mode.IDLE

        # ==================== KONDISI IDLE ====================
        if current_mode == Mode.IDLE:
            cv2.putText(img, "MODE: IDLE", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
            gesture_timer = None # Reset timer saat di IDLE

            # TELUNJUK LURUS = MASUK MODE TRACKING
            if fingers == [0, 1, 0, 0, 0]:
                current_mode = Mode.TRACKING

        # ==================== KONDISI TRACKING ====================
        elif current_mode == Mode.TRACKING:
            cv2.putText(img, "MODE: TRACKING", (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            # --- SUB-MODE: SCROLLING ---
            # Prioritaskan gestur scroll (telunjuk + tengah berdempetan).
            # Gestur ini aktif di seluruh area kamera, tidak dibatasi frameR.
            if fingers == [0, 1, 1, 0, 0]:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    cv2.putText(img, "SCROLLING", (wCam // 2 - 120, hCam // 2), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 255), cv2.FILLED)

                    current_y = lineInfo[5]

                    # Inisialisasi posisi awal scroll jika ini frame pertama
                    if last_scroll_y is None:
                        last_scroll_y = current_y

                    delta_y = current_y - last_scroll_y
                    scroll_amount = int(delta_y / SCROLL_SENSITIVITY) # buat negatif jika ingin scroll mengikuti gerakan tangan

                    if abs(scroll_amount) > 0:
                        pyautogui.scroll(scroll_amount)

                    last_scroll_y = current_y
                else:
                    # Jari merenggang, reset state scroll agar tidak ada lompatan saat gestur diaktifkan kembali
                    last_scroll_y = None

            # --- SUB-MODE: MOUSE MOVEMENT & CLICK ---
            # Hanya aktif jika gestur adalah jari telunjuk saja.
            # Kondisi: Telunjuk lurus, TAPI Jari Tengah & Manis ditekuk.
            # Ini memungkinkan Jempol atau Kelingking untuk bebas bergerak untuk klik
            # tanpa keluar dari mode TRACKING.
            # Gestur yang valid: [0,1,0,0,0], [1,1,0,0,0], [0,1,0,0,1], dll.
            elif fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                 # Reset state scroll saat beralih ke mode gerak mouse
                last_scroll_y = None

                x1, y1 = lmList[8][1:] # Ujung jari telunjuk

                # Konversi koordinat dari area kamera ke area layar
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Haluskan nilai koordinat
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Gerakkan kursor mouse
                autopy.mouse.move(clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

                # Logika Klik Kiri (single, double, hold)
                if not hasattr(detector, 'thumb_events'):
                    detector.thumb_events = []
                    detector.thumb_last_state = None
                    detector.thumb_hold_triggered = False

                thumb_folded = fingers[0] == 1
                current_time = time.time()

                # Catat perubahan status jempol
                if detector.thumb_last_state != thumb_folded:
                    detector.thumb_events.append((current_time, thumb_folded))
                    # Hapus event lebih dari 1 detik yang lalu
                    detector.thumb_events = [e for e in detector.thumb_events if current_time - e[0] <= 1.0]
                    detector.thumb_last_state = thumb_folded
                    detector.thumb_hold_triggered = False

                # Deteksi double click: dua kali tutup-buka dalam 1 detik
                thumb_closes = [e for e in detector.thumb_events if e[1]]
                thumb_opens = [e for e in detector.thumb_events if not e[1]]
                if len(thumb_closes) >= 2 and len(thumb_opens) >= 2:
                    # Pastikan urutan: close, open, close, open
                    idx = 0
                    pattern = []
                    for t, state in detector.thumb_events:
                        pattern.append(state)
                    if pattern[-4:] == [True, False, True, False]:
                        autopy.mouse.click(autopy.mouse.Button.LEFT)
                        autopy.mouse.click(autopy.mouse.Button.LEFT)
                        text = "Double Klik"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 3, 3)
                        x = max(0, (wCam - text_width) // 2)
                        y = max(text_height + 10, (hCam // 2))
                        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                        detector.thumb_events = []
                        time.sleep(0.3)
                        text = "Klik Kiri"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 3, 3)
                        x = max(0, (wCam - text_width) // 2)
                        y = max(text_height + 10, (hCam // 2))
                        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                elif len(thumb_closes) >= 1 and len(thumb_opens) >= 1:
                    if detector.thumb_events[-2:][0][1] and not detector.thumb_events[-2:][1][1]:
                        autopy.mouse.click(autopy.mouse.Button.LEFT)
                        cv2.putText(img, "Klik Kiri", (wCam // 2 - 120, hCam // 2), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                        detector.thumb_events = []
                        time.sleep(0.3)
                # Deteksi hold: jempol tetap terangkat selama 1 detik
                elif thumb_folded and not detector.thumb_hold_triggered:
                    text = "Tahan Kiri"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 3, 3)
                    x = max(0, (wCam - text_width) // 2)
                    y = max(text_height + 10, (hCam // 2))
                    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    for t, state in reversed(detector.thumb_events):
                        if state:
                            first_fold_time = t
                        else:
                            break
                    if first_fold_time and (current_time - first_fold_time >= 1.0):
                        autopy.mouse.toggle(autopy.mouse.Button.LEFT, down=True)
                        cv2.putText(img, "Tahan Kiri", (wCam // 2 - 120, hCam // 2), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                        detector.thumb_hold_triggered = True
                elif not thumb_folded:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, down=False)

                # Logika Klik Kanan
                if fingers[4] == 1:
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                    cv2.putText(img, "Klik Kanan", (wCam // 2 - 120, hCam // 2), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    time.sleep(0.5)

            else:
                # Jika gestur tidak valid untuk mode TRACKING (misal: jari telunjuk ditekuk), kembali ke IDLE
                current_mode = Mode.IDLE
                last_scroll_y = None # Reset state scroll saat keluar mode
    else: # Jika tidak ada tangan terdeteksi, kembali ke mode IDLE
        current_mode = Mode.IDLE

    # Menghitung dan menampilkan Frame Rate (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, str(int(fps)),  # Teks FPS
        (20, 50),  # Posisi teks
        cv2.FONT_HERSHEY_PLAIN,  # Font
        3,
        (255, 0, 255),
        3
    )

    # 12. Menampilkan gambar ke jendela
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # Menunggu 1ms, penting untuk menampilkan jendela GUI