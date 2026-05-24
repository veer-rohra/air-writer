<div align="center">

# ✍️ Air Writer

### Draw in the air using hand gestures  
### Powered by OpenCV + MediaPipe


<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-purple?style=for-the-badge&logo=google)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

![Stars](https://img.shields.io/github/stars/veer-rohra/air-writer?style=for-the-badge)
![Forks](https://img.shields.io/github/forks/veer-rohra/air-writer?style=for-the-badge)

</div>

---

# 📖 Project Overview

**Air Writer** is a real-time hand gesture drawing application that allows users to draw digitally in the air using only their hands and a webcam.

Using **MediaPipe Hand Tracking** and **OpenCV**, the project detects finger gestures and converts them into smooth drawing actions on a virtual canvas.

It can be used for:

- 🎨 Creative drawing
- 🧑‍🏫 Teaching & presentations
- ♿ Accessibility-based interaction
- 🖥️ Interactive demos
- 🧠 Computer vision learning

> **No mouse. No stylus. Just gestures.**

---

# ✨ Features

## 🎯 Core Functionalities

- ✋ Real-time hand tracking
- ✍️ Smooth air drawing
- 🎨 Multiple color options
- 🧼 Gesture-based eraser
- ⏸️ Pause drawing gesture
- 🤏 Dynamic brush size control
- 💾 Save artwork functionality
- 👋 Two-hand interaction support
- 🌈 Radial color picker

---

# 🚀 Versions

| Version | Features | Status |
|----------|----------|--------|
| **v1** | Basic drawing + color selection | ✅ Complete |
| **v2** | Radial color picker + stroke history | ✅ Complete |
| **v3** | Two-hand support + save drawing + dynamic brush | 🚀 Advanced |

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|-------------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe Hands |
| Numerical Operations | NumPy |
| Rendering/UI | OpenCV HighGUI |

---

# 📦 Dependencies

```txt
opencv-python>=4.8.0
mediapipe==0.10.9
numpy>=1.24.0
```

---

# 📁 Project Structure

```bash
air-writer/
│
├── air_writer_1.py
├── air_writer_v2.py
│
├── docs/
│   ├── air_writer_v3.py
│   └── index.html
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# ⚙️ Installation

## 📌 Prerequisites

- Python 3.8+
- Webcam

---

## 🔧 Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/veer-rohra/air-writer.git
cd air-writer
```

### 2️⃣ Create Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🎮 Usage

## ▶️ Run Different Versions

### Basic Version

```bash
python air_writer_1.py
```

### Enhanced Version

```bash
python air_writer_v2.py
```

### Latest Version (Recommended)

```bash
python docs/air_writer_v3.py
```

---

# ✋ Gesture Controls

| Gesture | Action |
|----------|--------|
| ☝️ Index Finger Up | Draw |
| ✋ Open Palm | Eraser Mode |
| ✌️ Index + Middle Finger | Pause Drawing |
| 🤏 Pinch Gesture | Change Brush Size |
| ✊ Fist Hold | Open Color Picker |
| 👋 Two Hands | Advanced Interactions |

---

# ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit Application |
| `S` | Save Drawing (v3) |

---



> Replace placeholder images with actual project screenshots or GIFs.

---

# 🔮 Future Improvements

- [ ] Web version using MediaPipe.js
- [ ] Shape recognition system
- [ ] Undo / Redo gestures
- [ ] Multi-user collaboration
- [ ] SVG export support
- [ ] AI-based stroke smoothing
- [ ] Virtual whiteboard mode
- [ ] Gesture shortcuts

---

# 🤝 Contributing

Contributions are welcome!

## Steps

```bash
# Fork repository

# Create feature branch
git checkout -b feature/amazing-feature

# Commit changes
git commit -m "Add amazing feature"

# Push branch
git push origin feature/amazing-feature
```

Then create a Pull Request 🚀

---

# 📋 Contribution Guidelines

- Follow PEP 8
- Write clean & readable code
- Add comments where needed
- Test under different lighting conditions
- Keep UI responsive and smooth

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

---

# 👨‍💻 Author

## Veer Rohra

- GitHub → https://github.com/veer-rohra
- LinkedIn → https://linkedin.com/in/veer-rohra

---

<div align="center">

## ⭐ If you like this project, consider starring the repository!

### Made with ❤️ using Python & Computer Vision

</div>
