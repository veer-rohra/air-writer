import cv2
import mediapipe as mp
import numpy as np
import math
import time

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

prev_x, prev_y = None, None
draw_color     = (0, 255, 0)
brush_thick    = 6
eraser_thick   = 80

FADE_DURATION  = 3.0   # seconds until a stroke is fully gone

# Each segment: {x1,y1,x2,y2, color, thick, birth}
segments = []

# ── Color palette [BGR] ──────────────────────────────────────────────────────
COLORS = [
    ("Green",  (0, 255,   0)),
    ("Cyan",   (255, 255,  0)),
    ("Blue",   (255, 100,  0)),
    ("Violet", (220,  50, 180)),
    ("Red",    (0,   0, 255)),
    ("Orange", (0, 165, 255)),
    ("Yellow", (0, 255, 255)),
    ("White",  (255, 255, 255)),
]

# ── Radial picker state ──────────────────────────────────────────────────────
picker_active  = False
picker_cx, picker_cy = 0, 0
fist_start     = None
FIST_HOLD_TIME = 0.7   # hold fist this long to open picker
PICKER_RADIUS  = 95


# ── Helpers ──────────────────────────────────────────────────────────────────
def landmark_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def fingers_up(lms):
    tips   = [4,  8, 12, 16, 20]
    joints = [3,  6, 10, 14, 18]
    up = [lms[tips[0]].x < lms[joints[0]].x]
    for i in range(1, 5):
        up.append(lms[tips[i]].y < lms[joints[i]].y)
    return up

def is_palm_open(lms):  return all(fingers_up(lms))
def is_two_fingers(lms):
    f = fingers_up(lms)
    return f[1] and f[2] and not f[3] and not f[4]
def is_fist(lms):
    f = fingers_up(lms)
    return not f[1] and not f[2] and not f[3] and not f[4]


# ── Glow renderer ────────────────────────────────────────────────────────────
def render_glow_segments(frame, segs, now):
    """
    Draw all segments with a neon glow + fading alpha.
    Strategy: paint onto a float32 glow layer, blur it, then add to frame.
    """
    if not segs:
        return frame

    glow_layer = np.zeros((H, W, 3), dtype=np.float32)
    core_layer = np.zeros((H, W, 3), dtype=np.float32)

    for seg in segs:
        age   = now - seg['birth']
        alpha = max(0.0, 1.0 - age / FADE_DURATION)
        if alpha <= 0:
            continue

        p1   = (seg['x1'], seg['y1'])
        p2   = (seg['x2'], seg['y2'])
        col  = tuple(c * alpha for c in seg['color'])  # fade colour
        t    = seg['thick']

        # Thick outer glow line
        cv2.line(glow_layer, p1, p2, col, t + 10, cv2.LINE_AA)
        # Bright mid line
        cv2.line(glow_layer, p1, p2, col, t + 3,  cv2.LINE_AA)
        # White hot core
        white = (255 * alpha, 255 * alpha, 255 * alpha)
        cv2.line(core_layer, p1, p2, white, max(1, t // 3), cv2.LINE_AA)

    # Blur to spread the glow
    blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=8, sigmaY=8)

    # Combine: frame + glow bloom + sharp mid + white core
    out = frame.astype(np.float32)
    out = np.clip(out + blurred * 0.9 + glow_layer * 0.6 + core_layer * 0.7, 0, 255)
    return out.astype(np.uint8)


# ── Radial colour picker ─────────────────────────────────────────────────────
def draw_radial_picker(img, cx, cy, selected):
    n = len(COLORS)

    # Dark backdrop circle
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), PICKER_RADIUS + 28, (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.circle(img, (cx, cy), PICKER_RADIUS + 28, (60, 60, 100), 1)

    for i, (name, col) in enumerate(COLORS):
        angle = (i / n) * 2 * math.pi - math.pi / 2
        scx   = int(cx + math.cos(angle) * PICKER_RADIUS)
        scy   = int(cy + math.sin(angle) * PICKER_RADIUS)
        active = (col == selected)
        r = 20 if active else 15

        # Glow halo
        cv2.circle(img, (scx, scy), r + 7, col, -1)
        blurred_spot = img.copy()
        cv2.GaussianBlur(blurred_spot, (15, 15), 5)
        # Solid swatch
        cv2.circle(img, (scx, scy), r, col, -1)
        if active:
            cv2.circle(img, (scx, scy), r + 3, (255, 255, 255), 2)
        cv2.putText(img, name[0], (scx - 5, scy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # Centre dot (shows current colour)
    cv2.circle(img, (cx, cy), 10, selected, -1)
    cv2.circle(img, (cx, cy), 12, (255, 255, 255), 1)
    cv2.putText(img, "color", (cx - 18, cy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def get_picker_hit(cx, cy, fx, fy):
    """Return colour BGR if fingertip is hovering over a swatch, else None."""
    for i, (_, col) in enumerate(COLORS):
        angle = (i / len(COLORS)) * 2 * math.pi - math.pi / 2
        scx   = int(cx + math.cos(angle) * PICKER_RADIUS)
        scy   = int(cy + math.sin(angle) * PICKER_RADIUS)
        if math.hypot(fx - scx, fy - scy) < 26:
            return col
    return None


# ── HUD ──────────────────────────────────────────────────────────────────────
def draw_ui(img, color, mode_text):
    # Mode badge (top-right)
    badge_col = {
        "DRAW":   (0, 140, 255),
        "ERASE":  (0,  50, 200),
        "PAUSE":  (80,  80, 80),
        "PICKER": (180, 80,   0),
    }.get(mode_text, (50, 50, 50))
    cv2.rectangle(img, (W - 158, 8), (W - 8, 42), badge_col, -1)
    cv2.putText(img, mode_text, (W - 153, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Current-colour swatch (top-left)
    cv2.circle(img, (30, 30), 16, color, -1)
    cv2.circle(img, (30, 30), 16, (255, 255, 255), 1)
    # Tiny glow ring
    glow_col = tuple(max(0, c - 60) for c in color)
    cv2.circle(img, (30, 30), 22, glow_col, 2)

    # Bottom hint bar
    hint = ("1-finger: Draw  |  2-finger: Pause  |  "
            "Fist (hold): Color Picker  |  Open palm: Erase  |  Q: Quit  |  C: Clear")
    cv2.rectangle(img, (0, H - 30), (W, H), (18, 18, 28), -1)
    cv2.putText(img, hint, (8, H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1, cv2.LINE_AA)


# ── Main loop ────────────────────────────────────────────────────────────────
print("Air Writer v2  |  Q = quit  |  C = clear")
print("NEW: Hold a FIST to open the radial colour picker.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)
    now   = time.time()

    mode_text = "IDLE"

    # ── Prune fully faded segments ─────────────────────────────────────────
    segments = [s for s in segments if now - s['birth'] < FADE_DURATION]

    # ── Render glowing fading strokes ──────────────────────────────────────
    frame = render_glow_segments(frame, segments, now)

    # ── Hand tracking ──────────────────────────────────────────────────────
    if res.multi_hand_landmarks:
        lm_list = res.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(
            frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec((0, 220, 255), 1, 2),
            mp_draw.DrawingSpec((255, 180, 0), 1, 1),
        )

        fx, fy = landmark_px(lm_list[8], W, H)   # index fingertip
        wx, wy = landmark_px(lm_list[0], W, H)   # wrist

        # ── Radial picker active ──
        if picker_active:
            mode_text  = "PICKER"
            hit        = get_picker_hit(picker_cx, picker_cy, fx, fy)
            if hit:
                draw_color = hit
            if not is_fist(lm_list):              # release fist → close picker
                picker_active = False
                fist_start    = None

        # ── Eraser ──
        elif is_palm_open(lm_list):
            mode_text  = "ERASE"
            fist_start = None
            # Remove segments whose midpoint is inside eraser radius
            er = eraser_thick // 2
            segments = [s for s in segments
                        if math.hypot((s['x1'] + s['x2']) / 2 - fx,
                                      (s['y1'] + s['y2']) / 2 - fy) > er]
            cv2.circle(frame, (fx, fy), er, (200, 200, 200), 2)
            prev_x, prev_y = None, None

        # ── Fist → open picker after hold ──
        elif is_fist(lm_list):
            mode_text  = "IDLE"
            prev_x, prev_y = None, None
            if fist_start is None:
                fist_start = now
            hold = now - fist_start
            # Draw a progress ring above the wrist
            sweep = int(360 * min(hold / FIST_HOLD_TIME, 1.0))
            cv2.ellipse(frame, (wx, wy - 35), (24, 24),
                        -90, 0, sweep, (0, 220, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "hold...", (wx - 26, wy - 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
            if hold >= FIST_HOLD_TIME:
                picker_active        = True
                picker_cx, picker_cy = wx, wy

        # ── Pen lift (2 fingers) ──
        elif is_two_fingers(lm_list):
            mode_text  = "PAUSE"
            fist_start = None
            cv2.circle(frame, (fx, fy), 10, (200, 200, 200), 2)
            prev_x, prev_y = None, None

        # ── Draw with index finger ──
        else:
            fist_start = None
            f = fingers_up(lm_list)
            if f[1]:
                mode_text = "DRAW"
                cv2.circle(frame, (fx, fy), brush_thick, draw_color, -1)
                if prev_x is not None:
                    segments.append({
                        'x1': prev_x, 'y1': prev_y,
                        'x2': fx,     'y2': fy,
                        'color': draw_color,
                        'thick': brush_thick,
                        'birth': now,
                    })
                prev_x, prev_y = fx, fy
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None
        if not picker_active:
            fist_start = None

    # ── Draw picker on top if active ──────────────────────────────────────
    if picker_active:
        draw_radial_picker(frame, picker_cx, picker_cy, draw_color)

    draw_ui(frame, draw_color, mode_text)

    cv2.imshow("✍  Air Writer v2  |  Q = quit  |  C = clear", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        segments.clear()
        print("Canvas cleared.")

cap.release()
cv2.destroyAllWindows()
