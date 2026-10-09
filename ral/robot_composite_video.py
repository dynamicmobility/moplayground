"""Tile the MuJoCo benchmark rollout videos into a single composite video.

Layout (default)::

    +-----------------+-----------------+-----------------+
    | cheetah  run    | walker   run    | ant      vx     |
    +-----------------+-----------------+-----------------+
    | cheetah  energy | walker   energy | ant      vy     |
    +--------+--------+--------+--------+-----------------+
    | humanoid run | humanoid energy | hopper run | hopper height |
    +--------------+-----------------+------------+---------------+

Every source clip is cropped to a region of interest (ROI) around the robot,
the ROI is grown (never distorted) to match its tile's aspect ratio, resized,
and pasted into the canvas. Frames are piped to ffmpeg (H.264, yuv420p) so the
result plays in browsers.

Usage::

    python ral/robot_composite_video.py                   # render the video
    python ral/robot_composite_video.py --preview 3.0     # save one frame (t=3s) as PNG
    python ral/robot_composite_video.py --measure         # print auto-detected robot boxes

Edit the constants below to change the layout, crops, labels, or output.
"""
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# Paths
# =============================================================================
VIDEO_DIR       = Path('docs/static/videos')
OUTPUT_PATH     = Path('ral/videos/robot_composite.mp4')
PREVIEW_PATH    = Path('ral/images/robot_composite_preview.png')

# =============================================================================
# Timing
# =============================================================================
SOURCE_FPS      = 100       # all benchmark clips are rendered at 100 fps
FRAME_STRIDE    = 2         # keep every Nth frame -> output fps = SOURCE_FPS / FRAME_STRIDE
START_TIME      = 0.0       # seconds into the source clips
END_TIME        = None      # seconds; None = until the shortest clip ends
PLAYBACK_SPEED  = 1.0       # >1 speeds the output up, <1 slows it down

# =============================================================================
# Canvas / layout
# =============================================================================
CANVAS_WIDTH    = 1920      # px; each row is split evenly among its panels
GAP             = 6         # px between tiles
OUTER_MARGIN    = 0         # px around the whole canvas
GAP_COLOR_BGR   = (170, 170, 170)   # separator / margin color

# Each row: (row height in px, [panels]). Each panel: (stack direction, [clip names]).
#   'vertical'   -> clips stacked top-to-bottom inside the panel
#   'horizontal' -> clips placed left-to-right inside the panel
LAYOUT = [
    (820, [
        ('vertical',   ['cheetah-run',  'cheetah-energy']),
        ('vertical',   ['walker-run',   'walker-energy']),
        ('vertical',   ['ant-vx',       'ant-vy']),
    ]),
    (580, [
        ('horizontal', ['humanoid-run', 'humanoid-energy']),
        ('horizontal', ['hopper-run',   'hopper-height']),
    ]),
]

# =============================================================================
# Crops
# =============================================================================
# Region of interest around each robot, in source pixels (x0, y0, x1, y1) on the
# 2560x1440 renders. The camera tracks the robot, so one ROI per robot covers
# both of its clips. The ROI is expanded around its center to match the tile's
# aspect ratio, so it only needs to bound the robot (+ shadow). Use --measure to
# see auto-detected bounds.
ROBOT_ROI = {
    'cheetah':  ( 740,  450, 1860, 1170),
    'walker':   ( 720,  210, 1690, 1330),
    'ant':      ( 890,  530, 1670, 1080),
    'humanoid': ( 780,  200, 1740, 1210),
    'hopper':   ( 830,  200, 1600, 1330),   # tall: camera tracks height, floor drifts
}
# Per-clip overrides, e.g. {'walker-energy': (1000, 210, 1600, 1330)}.
CLIP_ROI_OVERRIDES = {}
# Shift the expanded crop window by (dx, dy) source px, e.g. to show more floor.
CLIP_OFFSETS = {}

# =============================================================================
# Labels
# =============================================================================
SHOW_LABELS     = True
LABEL_FONT      = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
LABEL_SIZE      = 26        # px
LABEL_COLOR     = (40, 40, 40)      # RGB
LABEL_POS       = (14, 10)          # px from the tile's top-left corner
LABELS = {
    'cheetah-run':      'Cheetah · run',
    'cheetah-energy':   'Cheetah · energy',
    'walker-run':       'Walker · run',
    'walker-energy':    'Walker · energy',
    'ant-vx':           'Ant · vx',
    'ant-vy':           'Ant · vy',
    'humanoid-run':     'Humanoid · run',
    'humanoid-energy':  'Humanoid · energy',
    'hopper-run':       'Hopper · run',
    'hopper-height':    'Hopper · height',
}

# =============================================================================
# Encoding
# =============================================================================
CRF             = 20        # x264 quality (lower = better / bigger)
PRESET          = 'slow'
RESIZE_INTERP   = cv2.INTER_AREA

# Robot-color mask (HSV) used only by --measure.
ROBOT_HSV_LO    = (5, 80, 60)
ROBOT_HSV_HI    = (30, 255, 255)


# =============================================================================
# Implementation
# =============================================================================
@dataclass
class Tile:
    clip: str
    x: int
    y: int
    w: int
    h: int


def even(n):
    return int(n) // 2 * 2


def build_tiles():
    """Turn LAYOUT into absolute tile rectangles and the canvas size."""
    tiles = []
    inner_w = CANVAS_WIDTH - 2 * OUTER_MARGIN
    y = OUTER_MARGIN
    for row_h, panels in LAYOUT:
        n = len(panels)
        panel_w = (inner_w - GAP * (n - 1)) / n
        for p, (direction, clips) in enumerate(panels):
            px = OUTER_MARGIN + p * (panel_w + GAP)
            k = len(clips)
            for i, clip in enumerate(clips):
                if direction == 'vertical':
                    th = (row_h - GAP * (k - 1)) / k
                    tiles.append(Tile(clip, round(px), round(y + i * (th + GAP)), round(panel_w), round(th)))
                elif direction == 'horizontal':
                    tw = (panel_w - GAP * (k - 1)) / k
                    tiles.append(Tile(clip, round(px + i * (tw + GAP)), round(y), round(tw), round(row_h)))
                else:
                    raise ValueError(f'unknown stack direction {direction!r}')
        y += row_h + GAP
    canvas_h = even(y - GAP + OUTER_MARGIN)
    return tiles, (even(CANVAS_WIDTH), canvas_h)


def crop_window(clip, tile, src_w, src_h):
    """ROI for `clip`, expanded to the tile's aspect ratio and clamped to the frame."""
    robot = clip.split('-')[0]
    x0, y0, x1, y1 = CLIP_ROI_OVERRIDES.get(clip, ROBOT_ROI[robot])
    dx, dy = CLIP_OFFSETS.get(clip, (0, 0))
    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
    w, h = x1 - x0, y1 - y0
    aspect = tile.w / tile.h
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if w > src_w or h > src_h:  # shrink uniformly if the window no longer fits
        s = min(src_w / w, src_h / h)
        w, h = w * s, h * s
    x0 = int(round(np.clip(cx - w / 2, 0, src_w - w)))
    y0 = int(round(np.clip(cy - h / 2, 0, src_h - h)))
    return x0, y0, x0 + int(round(w)), y0 + int(round(h))


def make_label_overlay(tiles, size):
    """RGBA label layer rendered once with PIL, alpha-blended onto every frame."""
    overlay = Image.new('RGBA', size, (0, 0, 0, 0))
    if not SHOW_LABELS:
        return None
    try:
        font = ImageFont.truetype(LABEL_FONT, LABEL_SIZE)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(overlay)
    for t in tiles:
        text = LABELS.get(t.clip, t.clip)
        draw.text((t.x + LABEL_POS[0], t.y + LABEL_POS[1]), text, font=font, fill=(*LABEL_COLOR, 255))
    rgba = np.asarray(overlay).astype(np.float32) / 255.0
    alpha = rgba[..., 3:4]
    color_bgr = rgba[..., 2::-1]
    return color_bgr * 255.0, alpha


class ClipReader:
    def __init__(self, clip, tile):
        self.path = VIDEO_DIR / f'{clip}.mp4'
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(self.path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.tile = tile
        self.window = crop_window(clip, tile, src_w, src_h)

    def seek(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def read(self, skip=0):
        for _ in range(skip):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            return None
        x0, y0, x1, y1 = self.window
        return cv2.resize(frame[y0:y1, x0:x1], (self.tile.w, self.tile.h), interpolation=RESIZE_INTERP)

    def close(self):
        self.cap.release()


def compose(readers, size, labels, skip=0):
    canvas = np.empty((size[1], size[0], 3), np.uint8)
    canvas[:] = GAP_COLOR_BGR
    for r in readers:
        img = r.read(skip)
        if img is None:
            return None
        t = r.tile
        canvas[t.y:t.y + t.h, t.x:t.x + t.w] = img
    if labels is not None:
        color, alpha = labels
        canvas = (canvas * (1 - alpha) + color * alpha).astype(np.uint8)
    return canvas


def open_readers():
    tiles, size = build_tiles()
    readers = [ClipReader(t.clip, t) for t in tiles]
    return readers, size, make_label_overlay(tiles, size)


def render():
    readers, size, labels = open_readers()
    start = int(round(START_TIME * SOURCE_FPS))
    last = min(r.n_frames for r in readers)
    if END_TIME is not None:
        last = min(last, int(round(END_TIME * SOURCE_FPS)))
    n_out = len(range(start, last, FRAME_STRIDE))
    out_fps = SOURCE_FPS / FRAME_STRIDE * PLAYBACK_SPEED
    for r in readers:
        r.seek(start)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}', '-r', f'{out_fps}',
        '-i', '-',
        '-c:v', 'libx264', '-preset', PRESET, '-crf', str(CRF),
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(OUTPUT_PATH),
    ]
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    print(f'Rendering {n_out} frames at {size[0]}x{size[1]}, {out_fps:g} fps -> {OUTPUT_PATH}')
    for i in range(n_out):
        frame = compose(readers, size, labels, skip=0 if i == 0 else FRAME_STRIDE - 1)
        if frame is None:
            break
        ffmpeg.stdin.write(frame.tobytes())
        if i % 50 == 0:
            print(f'  frame {i}/{n_out}', flush=True)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    for r in readers:
        r.close()
    print('Done.')


def preview(t):
    readers, size, labels = open_readers()
    for r in readers:
        r.seek(int(round(t * SOURCE_FPS)))
    frame = compose(readers, size, labels)
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PREVIEW_PATH), frame)
    for r in readers:
        r.close()
    print(f'Saved preview (t={t}s, {size[0]}x{size[1]}) -> {PREVIEW_PATH}')


def measure(sample_every=10):
    """Print the union bounding box of robot-colored pixels for every clip."""
    tiles, _ = build_tiles()
    for t in tiles:
        cap = cv2.VideoCapture(str(VIDEO_DIR / f'{t.clip}.mp4'))
        box = [np.inf, np.inf, -np.inf, -np.inf]
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % sample_every == 0:
                _, frame = cap.retrieve()
                mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), ROBOT_HSV_LO, ROBOT_HSV_HI)
                ys, xs = np.nonzero(mask)
                if len(xs):
                    box = [min(box[0], xs.min()), min(box[1], ys.min()),
                           max(box[2], xs.max()), max(box[3], ys.max())]
            i += 1
        cap.release()
        print(f'{t.clip:18s} robot bbox (x0, y0, x1, y1) = {tuple(int(v) for v in box)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preview', type=float, metavar='T', help='save a single composite frame at T seconds')
    parser.add_argument('--measure', action='store_true', help='print auto-detected robot bounding boxes')
    args = parser.parse_args()

    if args.measure:
        measure()
    elif args.preview is not None:
        preview(args.preview)
    else:
        render()
