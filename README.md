# Dodge Punch

`Dodge Punch` is a webcam boxing reflex game built with Python, OpenCV, and MediaPipe.

The game tracks your head and arms so you can:

- dodge jabs and hooks
- block punches with your guard
- hit pad-work targets

## How To Play

When you run the game, stand in front of your webcam so your face is clearly visible.

### Dodge mode

- Move your head out of red jab/hook danger zones.
- Yellow shapes are warnings.
- Use your forearms/hands to block incoming punches when your guard lines overlap them.
- If a live attack hits your head box, the round ends.

### Pad work mode

- A pad appears near head height.
- It grows from yellow to blue.
- Hit the live blue pad with your arm line before it expires.
- Successful hits add points.

### Both mode

- The game alternates between dodge and pad-work stages.

### Controls

- `q` quits the game
- Click `Restart` after a knockout
- Auto-restart can also be enabled in `settings.py`

## Installation

This project is currently set up for Python `3.10`.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate it

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running The Game

```bash
python main.py
```

## Dependencies

The main Python packages are:

- `opencv-python`
- `mediapipe`
- `numpy`

The MediaPipe pose model file used by the game is already included in:

- `models/pose_landmarker_lite.task`

## What Machine Can Run It

This game should run on most modern laptops and desktops with a webcam.

### Minimum practical setup

- Python `3.10`
- A webcam
- A machine that can run OpenCV camera capture and MediaPipe pose tracking
- 4 GB RAM or more

### Recommended setup

- macOS, Windows, or Linux
- 8 GB RAM or more
- A recent Intel, AMD, or Apple Silicon CPU
- A webcam that supports `720p`
- Good room lighting so face and arm detection stay stable

### Notes

- A camera may support `60 FPS` only at certain resolutions or formats.
- If the HUD shows `@30`, your current camera mode is running at 30 FPS.
- Better lighting usually improves both face tracking and arm blocking.

## Configuration

Most game settings live in:

- [settings.py](./settings.py)

Useful values you can change:

- `GAME_MODE`
- `TARGET_FPS`
- `FULLSCREEN_WINDOW`
- jab/hook sizing and timing values
- pad-work timing values
- camera and face-detection tuning values

## Project Structure

- [main.py](./main.py): main game loop
- [settings.py](./settings.py): tunable variables and mode helpers
- [entities.py](./entities.py): attack and pad classes
- [combos.py](./combos.py): combo and spawn builders
- [vision.py](./vision.py): webcam, face detection, pose tracking, arm blocking
- [ui.py](./ui.py): HUD and on-screen text drawing
- [game_utils.py](./game_utils.py): shared geometry and helper utilities

## Troubleshooting

### Webcam does not open

- Make sure no other app is using the camera
- Check that Python has camera permission in your OS settings

### Tracking feels unstable

- Stand farther back so your upper body fits in frame
- Increase room lighting
- Make sure your face is not heavily backlit

### FPS is lower than expected

- Lower the camera resolution in `settings.py`
- Reduce face/pose detection cost in `settings.py`
- Check the on-screen timing values for `Read`, `Face`, and `Pose`
