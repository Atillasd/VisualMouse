import cv2
import mediapipe as mp
import pyautogui
import math
import time

# MediaPipe Tasks API (v0.10+)
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

HAND_MODEL_PATH = r"C:/Users/Atilla/Desktop/VisualMouse/hand_landmarker.task"

# El bağlantı çizgisi indeksleri (21 nokta)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Ekran boyutları
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # Gecikmeyi kapat

# Cooldown (saniye) - üst üste basmayı önler
CLICK_COOLDOWN_LEFT  = 3.0   # başparmak (sol tık) için 3 saniyelik cooldown
CLICK_COOLDOWN_RIGHT = 0.5   # serçe parmak (sağ tık) için cooldown

# Son tıklama zamanları
last_left_click  = 0.0
last_right_click = 0.0

# Çift tıklama tespiti: başparmak 0.6 sn içinde 2 kez kalkarsa çift tık
DBL_CLICK_WINDOW  = 0.6    # iki kaldırma arası maks süre (saniye)
thumb_raise_count = 0       # kaç kez kaldırıldı
thumb_first_raise = 0.0     # ilk kaldırma zamanı
prev_thumb_up     = False   # önceki karedeki başparmak durumu

# Scroll modu (pozisyon tabanlı)
SCROLL_SPEED    = 15          # scroll hız çarpanı
SCROLL_DEADZONE = 0.10        # merkezden bu oranda sapma olana kadar scroll yapma (0.0-0.5)

# Mouse yumuşatma (EMA) — 0.0=çok yumuşak, 1.0=ham hareket
SMOOTH_FACTOR = 0.35        # biraz artırıldı: daha hızlı tepki
smooth_x = None
smooth_y = None

# ROI: kameranın ortasındaki bu oransal bölge ekrana eşlenir.
# Bölge küçüldükçe daha az el hareketi ile ekranı dolaşırsın.
ROI_LEFT   = 0.30  # sol kenar (kamera genişliğinin oranı)
ROI_RIGHT  = 0.70   # sağ kenar
ROI_TOP    = 0.25   # üst kenar
ROI_BOTTOM = 0.75   # alt kenar

# Dead zone: bu piksel eşiğinden küçük hareketler yok sayılır (titreşim filtresi)
DEAD_ZONE_PX = 3

# Hız eğrisi (acceleration): yavaş hareket=hassas, hızlı hareket=geniş
ACCEL_THRESHOLD = 25   # daha düşük eşik → hızlanma daha erken devreye girer
ACCEL_FACTOR    = 2.0  # hızlı hareketlerde daha güçlü çarpan


def landmark_distance(lm1, lm2, frame_w, frame_h):
    """İki landmark arasındaki piksel mesafesini döndürür."""
    x1, y1 = lm1.x * frame_w, lm1.y * frame_h
    x2, y2 = lm2.x * frame_w, lm2.y * frame_h
    return math.hypot(x2 - x1, y2 - y1)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status_text = ""

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Piksel koordinatlarına çevir
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                # Bağlantı çizgileri
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, pts[start], pts[end], (0, 200, 255), 2)

                # Noktalar
                for pt in pts:
                    cv2.circle(frame, pt, 5, (255, 255, 255), -1)

                # ── ROI sınır kontrolü: el ROI dışındaysa jest algılama ──
                track_point = hand_landmarks[9]  # orta MCP
                in_roi = (ROI_LEFT <= track_point.x <= ROI_RIGHT and
                          ROI_TOP  <= track_point.y <= ROI_BOTTOM)

                if not in_roi:
                    smooth_x = None
                    smooth_y = None
                    cv2.putText(
                        frame,
                        "ROI DISI - El algilanmiyor",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )
                    continue  # bu el için jest algılamayı atla

                # Önemli noktaları vurgula
                # Başparmak ucu = 4, İşaret = 8, Orta = 12
                thumb_tip  = hand_landmarks[4]
                index_tip  = hand_landmarks[8]
                middle_tip = hand_landmarks[12]

                # ── El referans uzunluğu (bilek→orta parmak tabanı) ──
                wrist      = hand_landmarks[0]
                mid_base   = hand_landmarks[9]
                hand_size  = landmark_distance(wrist, mid_base, w, h)
                hand_size  = max(hand_size, 1)  # sıfıra bölmeyi önle

                # ── Parmak açık/kapalı tespiti ──
                def is_up(tip_i, pip_i):
                    return hand_landmarks[tip_i].y < hand_landmarks[pip_i].y

                index_up  = is_up(8,  6)
                middle_up = is_up(12, 10)
                ring_up   = is_up(16, 14)
                pinky_up  = is_up(20, 18)

                # Başparmak: ucu ile işaret MCP (5) arası mesafe > hand_size'ın yarısı
                dist_thumb = landmark_distance(hand_landmarks[4], hand_landmarks[5], w, h)
                thumb_up   = dist_thumb > hand_size * 0.5

                # Yumruk: işaret, orta, yüzük kapalı (serçe ve başparmak tıklama için serbest)
                is_fist = (not index_up and not middle_up
                           and not ring_up)

                # Scroll modu: işaret + orta parmak açık, yüzük ve serçe kapalı
                is_scroll = (index_up and middle_up
                             and not ring_up and not pinky_up)

                now = time.time()

                if is_scroll:
                    # ── SCROLL MODU (pozisyon tabanlı) ──
                    # Elin Y konumu: 0.0=üst, 1.0=alt, 0.5=orta
                    hand_y = hand_landmarks[9].y
                    offset = hand_y - 0.5  # negatif=üst, pozitif=alt

                    if abs(offset) > SCROLL_DEADZONE:
                        # Dead zone dışındaki kısmı orantıla (0→1 arası)
                        sign = 1 if offset > 0 else -1
                        strength = (abs(offset) - SCROLL_DEADZONE) / (0.5 - SCROLL_DEADZONE)
                        strength = min(strength, 1.0)
                        scroll_amount = int(sign * strength * SCROLL_SPEED)
                        if scroll_amount != 0:
                            pyautogui.scroll(-scroll_amount)  # eksi: aşağı el = aşağı scroll

                    # Ekranda mod göster
                    cv2.putText(
                        frame,
                        "SCROLL MODU",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 165, 0), 2
                    )
                    # Scroll modunda mouse smooth sıfırla
                    smooth_x = None
                    smooth_y = None

                elif is_fist:

                    # ── Mouse hareketi — orta MCP (9) ile takip ──
                    track = hand_landmarks[9]
                    raw_x = (track.x - ROI_LEFT)  / (ROI_RIGHT  - ROI_LEFT)
                    raw_y = (track.y - ROI_TOP)   / (ROI_BOTTOM - ROI_TOP)
                    raw_x = max(0.0, min(1.0, raw_x))
                    raw_y = max(0.0, min(1.0, raw_y))

                    target_x = raw_x * screen_width
                    target_y = raw_y * screen_height

                    if smooth_x is None:
                        smooth_x, smooth_y = target_x, target_y

                    # Hız eğrisi: delta büyükse çarpan uygula
                    dx = target_x - smooth_x
                    dy = target_y - smooth_y
                    dist = math.hypot(dx, dy)

                    if dist > ACCEL_THRESHOLD:
                        accel = 1.0 + (ACCEL_FACTOR - 1.0) * min(dist / 200.0, 1.0)
                        target_x = smooth_x + dx * accel
                        target_y = smooth_y + dy * accel
                        target_x = max(0, min(screen_width, target_x))
                        target_y = max(0, min(screen_height, target_y))

                    smooth_x = SMOOTH_FACTOR * target_x + (1 - SMOOTH_FACTOR) * smooth_x
                    smooth_y = SMOOTH_FACTOR * target_y + (1 - SMOOTH_FACTOR) * smooth_y

                    mouse_x = int(smooth_x)
                    mouse_y = int(smooth_y)

                    # Dead zone: çok küçük hareketleri yok say
                    if dist > DEAD_ZONE_PX:
                        pyautogui.moveTo(mouse_x, mouse_y, duration=0)

                    cv2.putText(
                        frame,
                        f"Mouse: ({mouse_x}, {mouse_y})",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2
                    )

                    # ── Başparmak → SOL TIK / ÇİFT TIK ──
                    if thumb_up and not prev_thumb_up:
                        # Başparmak yeni kalktı (kenar tespiti)
                        if now - thumb_first_raise > DBL_CLICK_WINDOW:
                            # Yeni sayma başlat
                            thumb_raise_count = 1
                            thumb_first_raise = now
                        else:
                            thumb_raise_count += 1

                        if thumb_raise_count >= 2:
                            # Çift tıklama!
                            cv2.circle(frame, pts[4], 16, (0, 255, 255), -1)
                            pyautogui.doubleClick()
                            last_left_click = now
                            thumb_raise_count = 0
                            status_text = "CIFT TIK!"
                        elif now - last_left_click > CLICK_COOLDOWN_LEFT:
                            # Tek tıklama (cooldown sonrası)
                            cv2.circle(frame, pts[4], 12, (0, 255, 0), -1)
                            pyautogui.click()
                            last_left_click = now
                            status_text = "SOL TIK!"

                    prev_thumb_up = thumb_up

                    # ── Serçe parmak kalkar → SAĞ TIK ──
                    if pinky_up:
                        cv2.circle(frame, pts[20], 12, (0, 0, 255), -1)
                        if now - last_right_click > CLICK_COOLDOWN_RIGHT:
                            pyautogui.rightClick()
                            last_right_click = now
                            status_text = "SAG TIK!"

                else:
                    # Yumruk/scroll değil → mouse dondur, smooth sıfırla
                    smooth_x = None
                    smooth_y = None
                    cv2.putText(
                        frame,
                        "MOUSE: PASIF",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (80, 80, 80), 2
                    )

        # ── ROI alanını görselleştir ──
        roi_x1 = int(ROI_LEFT   * w)
        roi_y1 = int(ROI_TOP    * h)
        roi_x2 = int(ROI_RIGHT  * w)
        roi_y2 = int(ROI_BOTTOM * h)

        # ROI dışını karart (yarı-şeffaf siyah overlay)
        overlay = frame.copy()
        # Üst bant
        cv2.rectangle(overlay, (0, 0), (w, roi_y1), (0, 0, 0), -1)
        # Alt bant
        cv2.rectangle(overlay, (0, roi_y2), (w, h), (0, 0, 0), -1)
        # Sol bant
        cv2.rectangle(overlay, (0, roi_y1), (roi_x1, roi_y2), (0, 0, 0), -1)
        # Sağ bant
        cv2.rectangle(overlay, (roi_x2, roi_y1), (w, roi_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # ROI sınır çerçevesi (yeşil kesikli çizgi efekti)
        dash_len = 10
        # Üst ve alt yatay çizgiler
        for x in range(roi_x1, roi_x2, dash_len * 2):
            cv2.line(frame, (x, roi_y1), (min(x + dash_len, roi_x2), roi_y1), (0, 255, 0), 2)
            cv2.line(frame, (x, roi_y2), (min(x + dash_len, roi_x2), roi_y2), (0, 255, 0), 2)
        # Sol ve sağ dikey çizgiler
        for y in range(roi_y1, roi_y2, dash_len * 2):
            cv2.line(frame, (roi_x1, y), (roi_x1, min(y + dash_len, roi_y2)), (0, 255, 0), 2)
            cv2.line(frame, (roi_x2, y), (roi_x2, min(y + dash_len, roi_y2)), (0, 255, 0), 2)

        # ROI köşe vurguları
        corner_len = 15
        corners = [(roi_x1, roi_y1), (roi_x2, roi_y1), (roi_x1, roi_y2), (roi_x2, roi_y2)]
        for cx, cy in corners:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # ROI içinde çapraz desen çizgileri (cross-hatch)
        mid_x = (roi_x1 + roi_x2) // 2
        mid_y = (roi_y1 + roi_y2) // 2
        cv2.line(frame, (mid_x, roi_y1), (mid_x, roi_y2), (0, 255, 0), 1)  # dikey orta
        cv2.line(frame, (roi_x1, mid_y), (roi_x2, mid_y), (0, 255, 0), 1)  # yatay orta

        # ROI etiketi
        cv2.putText(frame, "ROI", (roi_x1 + 5, roi_y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Durum mesajı (tıklama gerçekleşince)
        if status_text:
            cv2.putText(
                frame,
                status_text,
                (w // 2 - 80, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 255), 3
            )

        # Açıklama kutusu
        cv2.rectangle(frame, (0, h - 55), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Yumruk=Mouse | Basparmak=Sol/Cift Tik | Serce=Sag Tik | V=Scroll | Q=Cik",
                    (8, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        cv2.imshow("Visual Mouse - El Takip", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()