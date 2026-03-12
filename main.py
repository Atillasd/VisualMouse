import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time
import os

# ═══════════════════════════════════════════════════
#  ORTAK AYARLAR
# ═══════════════════════════════════════════════════

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


def landmark_distance(lm1, lm2, frame_w, frame_h):
    """İki landmark arasındaki piksel mesafesini döndürür."""
    x1, y1 = lm1.x * frame_w, lm1.y * frame_h
    x2, y2 = lm2.x * frame_w, lm2.y * frame_h
    return math.hypot(x2 - x1, y2 - y1)


def create_landmarker_options():
    """Ortak HandLandmarker ayarlarını döndürür."""
    return HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )


# ═══════════════════════════════════════════════════
#  MOUSE MODU  (HandTracking.py)
# ═══════════════════════════════════════════════════

def run_mouse_mode():
    """El ile mouse kontrol modu."""

    # Ekran boyutları
    screen_width, screen_height = pyautogui.size()
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    # Cooldown (saniye)
    CLICK_COOLDOWN_LEFT  = 3.0
    CLICK_COOLDOWN_RIGHT = 0.5
    last_left_click  = 0.0
    last_right_click = 0.0

    # Çift tıklama tespiti
    DBL_CLICK_WINDOW  = 0.6
    thumb_raise_count = 0
    thumb_first_raise = 0.0
    prev_thumb_up     = False

    # Scroll
    SCROLL_SPEED    = 15
    SCROLL_DEADZONE = 0.10

    # Mouse yumuşatma (EMA)
    SMOOTH_FACTOR = 0.35
    smooth_x = None
    smooth_y = None

    # ROI
    ROI_LEFT   = 0.30
    ROI_RIGHT  = 0.70
    ROI_TOP    = 0.25
    ROI_BOTTOM = 0.75

    # Dead zone & hız eğrisi
    DEAD_ZONE_PX    = 3
    ACCEL_THRESHOLD = 25
    ACCEL_FACTOR    = 2.0

    options = create_landmarker_options()
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
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                    for start, end in HAND_CONNECTIONS:
                        cv2.line(frame, pts[start], pts[end], (0, 200, 255), 2)
                    for pt in pts:
                        cv2.circle(frame, pt, 5, (255, 255, 255), -1)

                    # ROI sınır kontrolü
                    track_point = hand_landmarks[9]
                    in_roi = (ROI_LEFT <= track_point.x <= ROI_RIGHT and
                              ROI_TOP  <= track_point.y <= ROI_BOTTOM)

                    if not in_roi:
                        smooth_x = None
                        smooth_y = None
                        cv2.putText(frame, "ROI DISI - El algilanmiyor",
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        continue

                    thumb_tip  = hand_landmarks[4]
                    index_tip  = hand_landmarks[8]
                    middle_tip = hand_landmarks[12]

                    wrist      = hand_landmarks[0]
                    mid_base   = hand_landmarks[9]
                    hand_size  = landmark_distance(wrist, mid_base, w, h)
                    hand_size  = max(hand_size, 1)

                    def is_up(tip_i, pip_i):
                        return hand_landmarks[tip_i].y < hand_landmarks[pip_i].y

                    index_up  = is_up(8,  6)
                    middle_up = is_up(12, 10)
                    ring_up   = is_up(16, 14)
                    pinky_up  = is_up(20, 18)

                    dist_thumb = landmark_distance(hand_landmarks[4], hand_landmarks[5], w, h)
                    thumb_up   = dist_thumb > hand_size * 0.5

                    is_fist = (not index_up and not middle_up and not ring_up)

                    is_scroll = (index_up and middle_up
                                 and not ring_up and not pinky_up)

                    now = time.time()

                    if is_scroll:
                        hand_y = hand_landmarks[9].y
                        offset = hand_y - 0.5

                        if abs(offset) > SCROLL_DEADZONE:
                            sign = 1 if offset > 0 else -1
                            strength = (abs(offset) - SCROLL_DEADZONE) / (0.5 - SCROLL_DEADZONE)
                            strength = min(strength, 1.0)
                            scroll_amount = int(sign * strength * SCROLL_SPEED)
                            if scroll_amount != 0:
                                pyautogui.scroll(-scroll_amount)

                        cv2.putText(frame, "SCROLL MODU",
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                        smooth_x = None
                        smooth_y = None

                    elif is_fist:
                        track = hand_landmarks[9]
                        raw_x = (track.x - ROI_LEFT)  / (ROI_RIGHT  - ROI_LEFT)
                        raw_y = (track.y - ROI_TOP)   / (ROI_BOTTOM - ROI_TOP)
                        raw_x = max(0.0, min(1.0, raw_x))
                        raw_y = max(0.0, min(1.0, raw_y))

                        target_x = raw_x * screen_width
                        target_y = raw_y * screen_height

                        if smooth_x is None:
                            smooth_x, smooth_y = target_x, target_y

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

                        if dist > DEAD_ZONE_PX:
                            pyautogui.moveTo(mouse_x, mouse_y, duration=0)

                        cv2.putText(frame, f"Mouse: ({mouse_x}, {mouse_y})",
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Başparmak → SOL TIK / ÇİFT TIK
                        if thumb_up and not prev_thumb_up:
                            if now - thumb_first_raise > DBL_CLICK_WINDOW:
                                thumb_raise_count = 1
                                thumb_first_raise = now
                            else:
                                thumb_raise_count += 1

                            if thumb_raise_count >= 2:
                                cv2.circle(frame, pts[4], 16, (0, 255, 255), -1)
                                pyautogui.doubleClick()
                                last_left_click = now
                                thumb_raise_count = 0
                                status_text = "CIFT TIK!"
                            elif now - last_left_click > CLICK_COOLDOWN_LEFT:
                                cv2.circle(frame, pts[4], 12, (0, 255, 0), -1)
                                pyautogui.click()
                                last_left_click = now
                                status_text = "SOL TIK!"

                        prev_thumb_up = thumb_up

                        # Serçe parmak → SAĞ TIK
                        if pinky_up:
                            cv2.circle(frame, pts[20], 12, (0, 0, 255), -1)
                            if now - last_right_click > CLICK_COOLDOWN_RIGHT:
                                pyautogui.rightClick()
                                last_right_click = now
                                status_text = "SAG TIK!"

                    else:
                        smooth_x = None
                        smooth_y = None
                        cv2.putText(frame, "MOUSE: PASIF",
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

            # ROI görselleştirme
            roi_x1 = int(ROI_LEFT   * w)
            roi_y1 = int(ROI_TOP    * h)
            roi_x2 = int(ROI_RIGHT  * w)
            roi_y2 = int(ROI_BOTTOM * h)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, roi_y1), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, roi_y2), (w, h), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, roi_y1), (roi_x1, roi_y2), (0, 0, 0), -1)
            cv2.rectangle(overlay, (roi_x2, roi_y1), (w, roi_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            dash_len = 10
            for x in range(roi_x1, roi_x2, dash_len * 2):
                cv2.line(frame, (x, roi_y1), (min(x + dash_len, roi_x2), roi_y1), (0, 255, 0), 2)
                cv2.line(frame, (x, roi_y2), (min(x + dash_len, roi_x2), roi_y2), (0, 255, 0), 2)
            for y in range(roi_y1, roi_y2, dash_len * 2):
                cv2.line(frame, (roi_x1, y), (roi_x1, min(y + dash_len, roi_y2)), (0, 255, 0), 2)
                cv2.line(frame, (roi_x2, y), (roi_x2, min(y + dash_len, roi_y2)), (0, 255, 0), 2)

            corners = [(roi_x1, roi_y1), (roi_x2, roi_y1), (roi_x1, roi_y2), (roi_x2, roi_y2)]
            for cx, cy in corners:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            mid_x = (roi_x1 + roi_x2) // 2
            mid_y = (roi_y1 + roi_y2) // 2
            cv2.line(frame, (mid_x, roi_y1), (mid_x, roi_y2), (0, 255, 0), 1)
            cv2.line(frame, (roi_x1, mid_y), (roi_x2, mid_y), (0, 255, 0), 1)

            cv2.putText(frame, "ROI", (roi_x1 + 5, roi_y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if status_text:
                cv2.putText(frame, status_text, (w // 2 - 80, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            cv2.rectangle(frame, (0, h - 55), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, "Yumruk=Mouse | Basparmak=Sol/Cift Tik | Serce=Sag Tik | V=Scroll | Q=Menu",
                        (8, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

            cv2.imshow("Visual Mouse - Mouse Modu", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════
#  ÇİZİM MODU  (DrawingMode.py)
# ═══════════════════════════════════════════════════

def run_drawing_mode():
    """El ile çizim modu."""

    # Renk paleti (BGR)
    COLORS = [
        (0, 0, 255),     # Kırmızı
        (0, 255, 0),     # Yeşil
        (255, 0, 0),     # Mavi
        (0, 255, 255),   # Sarı
        (255, 0, 255),   # Mor
        (255, 165, 0),   # Turuncu
    ]
    COLOR_NAMES = ["Kirmizi", "Yesil", "Mavi", "Sari", "Mor", "Turuncu"]

    # Çizim ayarları
    DRAW_THICKNESS_THIN  = 3
    DRAW_THICKNESS_THICK = 8
    SMOOTH_FACTOR = 0.45
    ERASER_RADIUS = 60

    # Cooldown
    COLOR_CHANGE_COOLDOWN = 0.8

    # Durum değişkenleri
    current_color_idx   = 0
    last_color_change   = 0.0
    smooth_x            = None
    smooth_y            = None
    prev_draw_pt        = None
    prev_thumb_up       = False

    options = create_landmarker_options()
    cap = cv2.VideoCapture(0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

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
            mode_text   = "BEKLENIYOR"
            mode_color  = (80, 80, 80)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                    for start, end in HAND_CONNECTIONS:
                        cv2.line(frame, pts[start], pts[end], (0, 200, 255), 2)
                    for pt in pts:
                        cv2.circle(frame, pt, 4, (255, 255, 255), -1)

                    wrist    = hand_landmarks[0]
                    mid_base = hand_landmarks[9]
                    hand_size = landmark_distance(wrist, mid_base, w, h)
                    hand_size = max(hand_size, 1)

                    def is_up(tip_i, pip_i):
                        return hand_landmarks[tip_i].y < hand_landmarks[pip_i].y

                    index_up  = is_up(8,  6)
                    middle_up = is_up(12, 10)
                    ring_up   = is_up(16, 14)
                    pinky_up  = is_up(20, 18)

                    dist_thumb = landmark_distance(hand_landmarks[4], hand_landmarks[5], w, h)
                    thumb_up   = dist_thumb > hand_size * 0.5

                    now = time.time()

                    thickness = DRAW_THICKNESS_THICK if pinky_up else DRAW_THICKNESS_THIN

                    # Jestler
                    is_fist = (not index_up and not middle_up
                               and not ring_up and not pinky_up and not thumb_up)

                    is_draw = (index_up and not middle_up and not ring_up)

                    is_cursor = (index_up and middle_up and not ring_up)

                    is_color_change = (thumb_up and not index_up and not middle_up
                                       and not ring_up and not pinky_up
                                       and not prev_thumb_up)

                    if is_fist:
                        eraser_x = int(hand_landmarks[9].x * w)
                        eraser_y = int(hand_landmarks[9].y * h)

                        cv2.circle(canvas, (eraser_x, eraser_y), ERASER_RADIUS, (0, 0, 0), -1)

                        cv2.circle(frame, (eraser_x, eraser_y), ERASER_RADIUS, (255, 255, 255), 2)
                        cv2.circle(frame, (eraser_x, eraser_y), 2, (255, 255, 255), -1)
                        cv2.putText(frame, "SILGI", (eraser_x - 18, eraser_y - ERASER_RADIUS - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        prev_draw_pt = None
                        smooth_x = None
                        smooth_y = None
                        mode_text  = "SILGI"
                        mode_color = (200, 200, 200)

                    elif is_color_change:
                        if now - last_color_change > COLOR_CHANGE_COOLDOWN:
                            current_color_idx = (current_color_idx + 1) % len(COLORS)
                            last_color_change = now
                            status_text = f"RENK: {COLOR_NAMES[current_color_idx]}"
                        mode_text  = "RENK DEGISTIR"
                        mode_color = COLORS[current_color_idx]
                        prev_draw_pt = None
                        smooth_x = None
                        smooth_y = None

                    elif is_cursor:
                        ix = int(hand_landmarks[8].x * w)
                        iy = int(hand_landmarks[8].y * h)
                        cv2.circle(frame, (ix, iy), 10, COLORS[current_color_idx], 2)
                        cv2.circle(frame, (ix, iy), 3, (255, 255, 255), -1)
                        mode_text  = "IMLEC"
                        mode_color = (255, 200, 0)
                        prev_draw_pt = None
                        smooth_x = None
                        smooth_y = None

                    elif is_draw:
                        raw_x = hand_landmarks[8].x * w
                        raw_y = hand_landmarks[8].y * h

                        if smooth_x is None:
                            smooth_x, smooth_y = raw_x, raw_y
                        smooth_x = SMOOTH_FACTOR * raw_x + (1 - SMOOTH_FACTOR) * smooth_x
                        smooth_y = SMOOTH_FACTOR * raw_y + (1 - SMOOTH_FACTOR) * smooth_y

                        draw_x = int(smooth_x)
                        draw_y = int(smooth_y)

                        if prev_draw_pt is not None:
                            cv2.line(canvas, prev_draw_pt, (draw_x, draw_y),
                                     COLORS[current_color_idx], thickness)

                        prev_draw_pt = (draw_x, draw_y)

                        cv2.circle(frame, (draw_x, draw_y), thickness + 4,
                                   COLORS[current_color_idx], -1)

                        mode_text  = f"CIZIM ({COLOR_NAMES[current_color_idx]})"
                        mode_color = COLORS[current_color_idx]

                    else:
                        prev_draw_pt = None
                        smooth_x = None
                        smooth_y = None

                    prev_thumb_up = thumb_up

            else:
                prev_draw_pt = None
                smooth_x = None
                smooth_y = None

            # Canvas overlay
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            mask = (mask > 0).astype(np.uint8) * 255
            frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
            combined = cv2.add(frame_bg, canvas_fg)
            frame = cv2.addWeighted(combined, 0.8, frame, 0.2, 0)

            # Renk paleti göstergesi
            palette_y = 10
            for i, color in enumerate(COLORS):
                px = 10 + i * 35
                py = palette_y
                if i == current_color_idx:
                    cv2.rectangle(frame, (px - 3, py - 3), (px + 28, py + 28), (255, 255, 255), 2)
                cv2.rectangle(frame, (px, py), (px + 25, py + 25), color, -1)

            thickness_now = DRAW_THICKNESS_THICK if (result.hand_landmarks and pinky_up) else DRAW_THICKNESS_THIN
            cv2.putText(frame, f"Kalinlik: {thickness_now}px",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.putText(frame, mode_text,
                        (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

            if status_text:
                cv2.putText(frame, status_text, (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 0), -1)
            cv2.putText(frame,
                        "Isaret=Ciz | V=Imlec | Yumruk=Silgi | Basparmak=Renk | Serce=Kalin | S=Kaydet | C=Temizle | Q=Menu",
                        (8, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 180), 1)

            cv2.imshow("Visual Mouse - Cizim Modu", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_dir = r"C:/Users/Atilla/Desktop/VisualMouse"
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"cizim_{timestamp}.png")
                cv2.imwrite(filename, canvas)
                print(f"Kaydedildi: {filename}")
            elif key == ord("c"):
                canvas[:] = 0
                print("Tuval temizlendi!")

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════
#  ANA MENÜ
# ═══════════════════════════════════════════════════

def show_menu():
    """Açılış menüsünü gösterir. Kullanıcı mod seçene kadar döngüde kalır."""

    MENU_WIDTH  = 500
    MENU_HEIGHT = 400

    while True:
        # Menü arka planı
        menu = np.zeros((MENU_HEIGHT, MENU_WIDTH, 3), dtype=np.uint8)

        # Gradient arka plan (koyu mavi → siyah)
        for y in range(MENU_HEIGHT):
            ratio = y / MENU_HEIGHT
            b = int(80 * (1 - ratio))
            g = int(40 * (1 - ratio))
            menu[y, :] = (b, g, 0)

        # Başlık
        cv2.putText(menu, "VISUAL MOUSE",
                    (95, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 255), 3)

        # Ayırıcı çizgi
        cv2.line(menu, (50, 100), (450, 100), (0, 150, 200), 2)

        # Mod seçenekleri
        cv2.putText(menu, "Mod Seciniz:",
                    (160, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Seçenek 1: Mouse Modu
        cv2.rectangle(menu, (60, 180), (440, 240), (40, 40, 40), -1)
        cv2.rectangle(menu, (60, 180), (440, 240), (0, 200, 255), 2)
        cv2.putText(menu, "[1]  Mouse Modu",
                    (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Seçenek 2: Çizim Modu
        cv2.rectangle(menu, (60, 260), (440, 320), (40, 40, 40), -1)
        cv2.rectangle(menu, (60, 260), (440, 320), (0, 255, 100), 2)
        cv2.putText(menu, "[2]  Cizim Modu",
                    (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

        # Çıkış bilgisi
        cv2.putText(menu, "Q = Cikis  |  Modda Q = Menuye Don",
                    (70, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        cv2.imshow("Visual Mouse - Menu", menu)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("1"):
            cv2.destroyAllWindows()
            run_mouse_mode()
        elif key == ord("2"):
            cv2.destroyAllWindows()
            run_drawing_mode()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════
#  GİRİŞ NOKTASI
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    show_menu()
