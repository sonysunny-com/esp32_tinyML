# ESP32-CAM TinyML Demo (Person Detection)

A minimal PlatformIO project for running **TinyML (TensorFlow Lite Micro)** on the **AI Thinker ESP32-CAM**.

It captures grayscale frames, runs an on-device `person_detection` model, and toggles the onboard **flash LED** when a person is detected.

## ⚙️ Setup
1. Board: AI Thinker ESP32-CAM (OV2640)
2. Programmer: FTDI USB-TTL (5V, GND, TX->U0R, RX->U0T)
3. Flash mode: Connect GPIO0 → GND during upload.

## 🔧 Build (PlatformIO)
```bash
pio run -t upload
pio device monitor
```

## 🧠 Replace the Model
Replace `src/person_detect_model_data.h` with the real header from the Arduino TensorFlow Lite `person_detection` example.

Then rebuild and re-upload.
