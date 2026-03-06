import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
from datetime import datetime

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,                  # ★ Two-hand support
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

# ── Constants ────────────────────────────────────────────────────────────────
FADE_DURATION    = 3.0    # seconds until a stroke is fully gone
FIST_HOLD_TIME   = 0.7    # hold fist this long to open picker
PICKER_RADIUS    = 95
DEFAULT_BRUSH    = 6
MIN_BRUSH        = 2
MAX_BRUSH        = 30
ERASER_RADIUS    = 40     # pixels (replaces magic number 80 // 2)
PINCH_MIN_DIST   = 20     # pixels – smallest pinch → smallest brush
PINCH_MAX_DIST   = 100    # pixels – widest pinch  → largest brush
PINCH_LOCK_DIST  = 40     # below this → treat as "pinching" (no draw)

# ── Color palette [BGR] ──────────────────────────────────────────────────────
COLORS = [
    ("Green",  (0,   255,   0)),
    ("Cyan",   (255, 255,   0)),
    ("Blue",   (255, 100,   0)),
    ("Violet", (220,  50, 180)),
    ("Red",    (0,     0, 255)),
    ("Orange", (0,   165, 255)),
    ("Yellow", (0,   255, 255)),
    ("White",  (255, 255, 255)),
]

# ── Camera init ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera. Check your camera connection.")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("ERROR: Could not read from camera.")
    cap.release()
    exit(1)

H, W = frame.shape[:2]

# ── Saves directory ───────────────────────────────────────────────────────────
SAVES_DIR = os.path.join(os.path.expanduser("~"), "air_writer_saves")
os.makedirs(SAVES_DIR, exist_ok=True)

# ── App state ────────────────────────────────────────────────────────────────
prev_x, prev_y  = None, None
draw_color      = (0, 255, 0)
brush_thick     = DEFAULT_BRUSH
fade_enabled    = True          # ★ Fade toggle
dark_bg         = True          # ★ Background toggle

# Stroke groups for undo: list-of-lists; each inner list = one continuous stroke
stroke_groups   = []
current_stroke  = []
segments        = []            # flat cache rebuilt from stroke_groups

# Radial picker state
picker_active        = False
picker_cx, picker_cy = 0, 0
fist_start           = None

# Save notification
save_notification = ""
save_notify_time  = 0.0


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

def pinch_distance(lms):
    """Pixel distance between thumb tip (4) and index tip (8)."""
    tx, ty = landmark_px(lms[4], W, H)
    ix, iy = landmark_px(lms[8], W, H)
    return math.hypot(tx - ix, ty - iy)

def is_pinching(lms):
    return pinch_distance(lms) < PINCH_LOCK_DIST

def brush_from_pinch(dist):
    """Map pinch spread to brush thickness."""
    t = (dist - PINCH_MIN_DIST) / (PINCH_MAX_DIST - PINCH_MIN_DIST)
    t = max(0.0, min(1.0, t))
    return int(MIN_BRUSH + t * (MAX_BRUSH - MIN_BRUSH))

def rebuild_segments():
    """Sync flat segments cache from stroke_groups."""
    global segments
    segments = [seg for group in stroke_groups for seg in group]

def finish_stroke():
    """Commit current_stroke as a new undo-able group."""
    global current_stroke
    if current_stroke:
        stroke_groups.append(current_stroke)
        current_stroke = []


# ── Glow renderer ────────────────────────────────────────────────────────────
def render_glow_segments(img, segs, now):
    if not segs:
        return img

    glow_layer = np.zeros((H, W, 3), dtype=np.float32)
    core_layer = np.zeros((H, W, 3), dtype=np.float32)

    for seg in segs:
        age   = now - seg['birth']
        alpha = max(0.0, 1.0 - age / FADE_DURATION) if fade_enabled else 1.0
        if alpha <= 0:
            continue

        p1  = (seg['x1'], seg['y1'])
        p2  = (seg['x2'], seg['y2'])
        col = tuple(c * alpha for c in seg['color'])
        t   = seg['thick']

        cv2.line(glow_layer, p1, p2, col, t + 10, cv2.LINE_AA)
        cv2.line(glow_layer, p1, p2, col, t + 3,  cv2.LINE_AA)
        white = (255 * alpha, 255 * alpha, 255 * alpha)
        cv2.line(core_layer, p1, p2, white, max(1, t // 3), cv2.LINE_AA)

    blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=8, sigmaY=8)
    out = img.astype(np.float32)
    out = np.clip(out + blurred * 0.9 + glow_layer * 0.6 + core_layer * 0.7, 0, 255)
    return out.astype(np.uint8)


# ── Radial colour picker ──────────────────────────────────────────────────────
def draw_radial_picker(img, cx, cy, selected):
    n = len(COLORS)
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), PICKER_RADIUS + 28, (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.circle(img, (cx, cy), PICKER_RADIUS + 28, (60, 60, 100), 1)

    for i, (name, col) in enumerate(COLORS):
        angle  = (i / n) * 2 * math.pi - math.pi / 2
        scx    = int(cx + math.cos(angle) * PICKER_RADIUS)
        scy    = int(cy + math.sin(angle) * PICKER_RADIUS)
        active = (col == selected)
        r      = 20 if active else 15

        cv2.circle(img, (scx, scy), r + 7, col, -1)   # glow halo
        cv2.circle(img, (scx, scy), r, col, -1)        # solid swatch
        if active:
            cv2.circle(img, (scx, scy), r + 3, (255, 255, 255), 2)
        cv2.putText(img, name[0], (scx - 5, scy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.circle(img, (cx, cy), 10, selected, -1)
    cv2.circle(img, (cx, cy), 12, (255, 255, 255), 1)
    cv2.putText(img, "color", (cx - 18, cy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

def get_picker_hit(cx, cy, fx, fy):
    for i, (_, col) in enumerate(COLORS):
        angle = (i / len(COLORS)) * 2 * math.pi - math.pi / 2
        scx   = int(cx + math.cos(angle) * PICKER_RADIUS)
        scy   = int(cy + math.sin(angle) * PICKER_RADIUS)
        if math.hypot(fx - scx, fy - scy) < 26:
            return col
    return None


# ── Gesture handlers ──────────────────────────────────────────────────────────
def handle_draw(img, lm_list, now):
    """Index finger extended → draw. Pinch spread controls brush size."""
    global prev_x, prev_y, brush_thick, current_stroke

    fx, fy = landmark_px(lm_list[8], W, H)
    f      = fingers_up(lm_list)
    dist   = pinch_distance(lm_list)

    # ★ Dynamic brush size from pinch spread
    if PINCH_MIN_DIST <= dist <= PINCH_MAX_DIST:
        brush_thick = brush_from_pinch(dist)

    if f[1] and not is_pinching(lm_list):
        cv2.circle(img, (fx, fy), brush_thick, draw_color, -1)
        if prev_x is not None:
            seg = {
                'x1': prev_x, 'y1': prev_y,
                'x2': fx,     'y2': fy,
                'color': draw_color,
                'thick': brush_thick,
                'birth': now,
            }
            current_stroke.append(seg)
            segments.append(seg)
        prev_x, prev_y = fx, fy
        return "DRAW"
    else:
        finish_stroke()
        prev_x, prev_y = None, None
        return "IDLE"


def handle_erase(img, lm_list):
    """Open palm → proximity eraser on draw hand."""
    global prev_x, prev_y
    fx, fy = landmark_px(lm_list[8], W, H)
    _erase_near(fx, fy)
    cv2.circle(img, (fx, fy), ERASER_RADIUS, (200, 200, 200), 2)
    prev_x, prev_y = None, None
    return "ERASE"


def _erase_near(fx, fy):
    """Remove segments whose midpoint falls within ERASER_RADIUS of (fx, fy)."""
    to_keep = []
    for group in stroke_groups:
        kept = [s for s in group
                if math.hypot((s['x1'] + s['x2']) / 2 - fx,
                               (s['y1'] + s['y2']) / 2 - fy) > ERASER_RADIUS]
        if kept:
            to_keep.append(kept)
    stroke_groups.clear()
    stroke_groups.extend(to_keep)
    rebuild_segments()


def handle_fist(img, lm_list, now):
    """Hold fist → progress ring → open colour picker."""
    global fist_start, picker_active, picker_cx, picker_cy, prev_x, prev_y
    wx, wy = landmark_px(lm_list[0], W, H)
    prev_x, prev_y = None, None
    if fist_start is None:
        fist_start = now
    hold  = now - fist_start
    sweep = int(360 * min(hold / FIST_HOLD_TIME, 1.0))
    cv2.ellipse(img, (wx, wy - 35), (24, 24),
                -90, 0, sweep, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "hold...", (wx - 26, wy - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    if hold >= FIST_HOLD_TIME:
        picker_active        = True
        picker_cx, picker_cy = wx, wy
    finish_stroke()
    return "IDLE"


def handle_picker(lm_list):
    """Picker active → track fingertip over swatches; release fist to close."""
    global draw_color, picker_active, fist_start
    fx, fy = landmark_px(lm_list[8], W, H)
    hit    = get_picker_hit(picker_cx, picker_cy, fx, fy)
    if hit:
        draw_color = hit
    if not is_fist(lm_list):
        picker_active = False
        fist_start    = None
    return "PICKER"


def handle_two_fingers(img, lm_list):
    """Two fingers up → pen lift / pause."""
    global prev_x, prev_y
    fx, fy = landmark_px(lm_list[8], W, H)
    cv2.circle(img, (fx, fy), 10, (200, 200, 200), 2)
    prev_x, prev_y = None, None
    finish_stroke()
    return "PAUSE"


def handle_control_hand(img, lm_list):
    """★ Second hand: open palm acts as a large eraser."""
    if is_palm_open(lm_list):
        fx, fy = landmark_px(lm_list[8], W, H)
        _erase_near(fx, fy)
        cv2.circle(img, (fx, fy), ERASER_RADIUS, (100, 100, 255), 2)
        cv2.putText(img, "Erase (L)", (fx - 36, fy - ERASER_RADIUS - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 255), 1, cv2.LINE_AA)


# ── Save canvas ───────────────────────────────────────────────────────────────
def save_canvas(img):
    fname = datetime.now().strftime("air_writer_%Y%m%d_%H%M%S.png")
    path  = os.path.join(SAVES_DIR, fname)
    cv2.imwrite(path, img)
    print(f"✅ Saved: {path}")
    return os.path.basename(path)


# ── Background overlay ────────────────────────────────────────────────────────
def apply_background(img, dark):
    """★ Darken camera feed for dark mode, or lighten for light mode."""
    if dark:
        overlay = np.zeros_like(img)
        return cv2.addWeighted(img, 0.35, overlay, 0.65, 0)
    else:
        overlay = np.full_like(img, 230)
        return cv2.addWeighted(img, 0.25, overlay, 0.75, 0)


# ── HUD ──────────────────────────────────────────────────────────────────────
def draw_ui(img, color, mode_text):
    # Mode badge (top-right)
    badge_col = {
        "DRAW":   (0, 140, 255),
        "ERASE":  (0,  50, 200),
        "PAUSE":  (80,  80,  80),
        "PICKER": (180, 80,   0),
    }.get(mode_text, (50, 50, 50))
    cv2.rectangle(img, (W - 158, 8), (W - 8, 42), badge_col, -1)
    cv2.putText(img, mode_text, (W - 153, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Toggle indicators
    fade_lbl = "Fade: ON " if fade_enabled else "Fade: OFF"
    bg_lbl   = "BG: Dark " if dark_bg      else "BG: Light"
    cv2.putText(img, fade_lbl, (W - 158, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1, cv2.LINE_AA)
    cv2.putText(img, bg_lbl, (W - 158, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 220), 1, cv2.LINE_AA)

    # Colour swatch + brush info (top-left)
    cv2.circle(img, (30, 30), 16, color, -1)
    cv2.circle(img, (30, 30), 16, (255, 255, 255), 1)
    glow_col = tuple(max(0, c - 60) for c in color)
    cv2.circle(img, (30, 30), 22, glow_col, 2)
    cv2.putText(img, f"Brush: {brush_thick}px", (55, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"Strokes: {len(stroke_groups)}", (55, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 180), 1, cv2.LINE_AA)

    # Bottom hint bar
    hint = ("1-finger: Draw  |  Pinch: Brush size  |  2-finger: Pause  |  "
            "Fist(hold): Color  |  Palm: Erase  |  "
            "Z:Undo  S:Save  F:Fade  B:BG  C:Clear  Q:Quit")
    cv2.rectangle(img, (0, H - 30), (W, H), (18, 18, 28), -1)
    cv2.putText(img, hint, (8, H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 200), 1, cv2.LINE_AA)


# ── Main loop ────────────────────────────────────────────────────────────────
print("Air Writer v3  |  Q=quit  C=clear  Z=undo  S=save  F=fade  B=bg")
print(f"Saves folder: {SAVES_DIR}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = apply_background(frame, dark_bg)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)
    now   = time.time()
    mode_text = "IDLE"

    # ── Prune faded segments ──────────────────────────────────────────────
    if fade_enabled:
        for group in stroke_groups:
            group[:] = [s for s in group if now - s['birth'] < FADE_DURATION]
        stroke_groups[:] = [g for g in stroke_groups if g]
        rebuild_segments()

    # ── Render strokes ────────────────────────────────────────────────────
    frame = render_glow_segments(frame, segments, now)

    # ── Hand tracking ─────────────────────────────────────────────────────
    if res.multi_hand_landmarks:
        draw_lms = None
        ctrl_lms = None

        if len(res.multi_hand_landmarks) == 1:
            # ★ Single hand: always treat as drawing hand
            draw_lms = res.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(
                frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec((0, 220, 255), 1, 2),
                mp_draw.DrawingSpec((255, 180, 0), 1, 1),
            )
        else:
            # ★ Two hands: right = draw, left = erase/control
            for i, hand_info in enumerate(res.multi_handedness):
                label = hand_info.classification[0].label
                lms   = res.multi_hand_landmarks[i].landmark
                mp_draw.draw_landmarks(
                    frame, res.multi_hand_landmarks[i], mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec((0, 220, 255), 1, 2),
                    mp_draw.DrawingSpec((255, 180, 0), 1, 1),
                )
                if label == "Right":
                    draw_lms = lms
                else:
                    ctrl_lms = lms

            if ctrl_lms:
                handle_control_hand(frame, ctrl_lms)

        # ── Process drawing hand ──────────────────────────────────────────
        if draw_lms:
            if picker_active:
                mode_text = handle_picker(draw_lms)
            elif is_palm_open(draw_lms):
                fist_start = None
                mode_text  = handle_erase(frame, draw_lms)
            elif is_fist(draw_lms):
                mode_text  = handle_fist(frame, draw_lms, now)
            elif is_two_fingers(draw_lms):
                fist_start = None
                mode_text  = handle_two_fingers(frame, draw_lms)
            else:
                fist_start = None
                mode_text  = handle_draw(frame, draw_lms, now)
    else:
        finish_stroke()
        prev_x, prev_y = None, None
        if not picker_active:
            fist_start = None

    # ── Picker overlay ────────────────────────────────────────────────────
    if picker_active:
        draw_radial_picker(frame, picker_cx, picker_cy, draw_color)

    # ── HUD ───────────────────────────────────────────────────────────────
    draw_ui(frame, draw_color, mode_text)

    # ── Save notification banner ──────────────────────────────────────────
    if save_notification and now - save_notify_time < 2.5:
        cv2.rectangle(frame, (W // 2 - 220, H // 2 - 25),
                              (W // 2 + 220, H // 2 + 15), (20, 60, 20), -1)
        cv2.putText(frame, f"✅  {save_notification}",
                    (W // 2 - 210, H // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

    cv2.imshow("✍  Air Writer v3  |  Q=quit  Z=undo  S=save", frame)

    # ── Keyboard shortcuts ────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        stroke_groups.clear()
        segments.clear()
        current_stroke.clear()
        print("Canvas cleared.")
    elif key == ord('z'):                      # ★ Undo
        if stroke_groups:
            removed = stroke_groups.pop()
            rebuild_segments()
            print(f"Undo: removed stroke ({len(removed)} segments)")
        else:
            print("Nothing to undo.")
    elif key == ord('s'):                      # ★ Save
        fname = save_canvas(frame)
        save_notification = fname
        save_notify_time  = now
    elif key == ord('f'):                      # ★ Fade toggle
        fade_enabled = not fade_enabled
        if not fade_enabled:
            for seg in segments:             # reset birth → stays visible
                seg['birth'] = now
        print(f"Fade {'ON' if fade_enabled else 'OFF'}")
    elif key == ord('b'):                      # ★ Background toggle
        dark_bg = not dark_bg
        print(f"Background: {'Dark' if dark_bg else 'Light'}")

cap.release()
cv2.destroyAllWindows()
