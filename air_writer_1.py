import cv2
import mediapipe as mp
import numpy as np
import math

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

# ── Canvas & state ───────────────────────────────────────────────────────────
cap          = cv2.VideoCapture(0)
ret, frame   = cap.read()
H, W         = frame.shape[:2]

canvas       = np.zeros((H, W, 3), dtype=np.uint8)   # persistent drawing layer
prev_x, prev_y = None, None                           # last draw position
draw_color   = (0, 255, 0)                            # default: green
brush_thick  = 6
eraser_thick = 80

# Color palette  [BGR]
COLORS = {
    "Green"  : (0, 255, 0),
    "Blue"   : (255, 100, 0),
    "Red"    : (0, 0, 255),
    "Yellow" : (0, 255, 255),
    "White"  : (255, 255, 255),
}
color_names = list(COLORS.keys())
color_vals  = list(COLORS.values())

# ── Helpers ──────────────────────────────────────────────────────────────────
def landmark_px(lm, w, h):
    """Convert normalised landmark to pixel coords."""
    return int(lm.x * w), int(lm.y * h)

def fingers_up(lms):
    """Return list of booleans [thumb, index, middle, ring, pinky]."""
    tips   = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    up = []
    # Thumb: compare x (horizontal flip handled by mirroring)
    up.append(lms[tips[0]].x < lms[joints[0]].x)
    # Other fingers: tip y < pip y  ⟹ finger extended
    for i in range(1, 5):
        up.append(lms[tips[i]].y < lms[joints[i]].y)
    return up

def is_palm_open(lms):
    """All four fingers + thumb extended → open palm = eraser."""
    return all(fingers_up(lms))

def is_two_fingers(lms):
    """Index + middle up, rest down → pen-lift / pause drawing."""
    f = fingers_up(lms)
    return f[1] and f[2] and not f[3] and not f[4]

def draw_ui(img, color, mode_text):
    """Overlay colour palette, mode label, and instructions."""
    # Colour swatches (top-left)
    sw, sh, gap = 40, 30, 6
    for i, (name, col) in enumerate(zip(color_names, color_vals)):
        x1 = gap + i * (sw + gap)
        y1 = gap
        cv2.rectangle(img, (x1, y1), (x1 + sw, y1 + sh), col, -1)
        if col == color:
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x1 + sw + 2, y1 + sh + 2),
                          (255, 255, 255), 2)
        cv2.putText(img, name[0], (x1 + 13, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Mode badge
    badge_col = (0, 140, 255) if "DRAW" in mode_text else \
                (0, 0, 200)   if "ERASE" in mode_text else (80, 80, 80)
    cv2.rectangle(img, (W - 160, gap), (W - gap, gap + 34), badge_col, -1)
    cv2.putText(img, mode_text, (W - 155, gap + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Bottom hint bar
    hint = "1-finger: Draw | 2-finger: Pause | Open palm: Erase | Q: Quit | C: Clear"
    cv2.rectangle(img, (0, H - 28), (W, H), (30, 30, 30), -1)
    cv2.putText(img, hint, (8, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

# ── Main loop ────────────────────────────────────────────────────────────────
print("Air Writer started.  Press Q to quit, C to clear canvas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)                        # mirror like a selfie
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    mode_text = "IDLE"

    if res.multi_hand_landmarks:
        lm_list = res.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(frame,
                               res.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec((0, 220, 255), 1, 2),
                               mp_draw.DrawingSpec((255, 180, 0), 1, 1))

        fx, fy = landmark_px(lm_list[8], W, H)        # index fingertip
        tx, ty = landmark_px(lm_list[4], W, H)        # thumb tip

        # ── Colour selection: touch thumb to a swatch ──
        sw, sh, gap = 40, 30, 6
        for i, col in enumerate(color_vals):
            x1 = gap + i * (sw + gap)
            y1 = gap
            if x1 <= tx <= x1 + sw and y1 <= ty <= y1 + sh:
                draw_color = col

        if is_palm_open(lm_list):
            # ERASE at index-finger position
            mode_text = "ERASE"
            cv2.circle(frame, (fx, fy), eraser_thick // 2,
                       (255, 255, 255), 2)
            cv2.circle(canvas, (fx, fy), eraser_thick // 2,
                       (0, 0, 0), -1)
            prev_x, prev_y = None, None

        elif is_two_fingers(lm_list):
            # PEN LIFT — just hover, no drawing
            mode_text = "PAUSE"
            cv2.circle(frame, (fx, fy), 10, (200, 200, 200), 2)
            prev_x, prev_y = None, None

        else:
            # DRAW with index finger
            f = fingers_up(lm_list)
            if f[1]:                                   # index finger up
                mode_text = "DRAW"
                cv2.circle(frame, (fx, fy), brush_thick,
                           draw_color, -1)
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (fx, fy),
                             draw_color, brush_thick)
                prev_x, prev_y = fx, fy
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    # ── Blend canvas onto live frame ──────────────────────────────────────
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask     = cv2.threshold(canvas_gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv    = cv2.bitwise_not(mask)
    bg          = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg          = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined    = cv2.add(bg, fg)

    draw_ui(combined, draw_color, mode_text)

    cv2.imshow("✍  Air Writer  |  Q = quit  |  C = clear", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0                                 # clear canvas
        print("Canvas cleared.")

cap.release()
cv2.destroyAllWindows()
