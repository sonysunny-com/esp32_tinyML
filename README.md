# ESP32-CAM TinyML Demo (Person Detection)

A minimal starter showing how to run **TinyML** (TensorFlow Lite Micro) on the **AI Thinker ESP32-CAM** by sonysunny.com

It captures grayscale frames, resizes to **96×96**, runs a **person detection** model,
and toggles the onboard **flash LED (GPIO4)** when a person is detected.

> This repo ships with a **placeholder model header** so it compiles out-of-the-box.
> Replace `src/person_detect_model_data.h` with the *real* header from the
> Arduino TensorFlow Lite **person_detection** example to enable inference.

---

## Hardware

- **Board:** AI Thinker ESP32-CAM (OV2640)
- **Programmer:** FTDI/USB-TTL (5V, GND, TX→U0R, RX→U0T)
- **Flash mode:** tie **GPIO0 → GND** during upload; remove after flashing and reset.

Power the board via **5V** (camera needs stable power).

---

## Build (PlatformIO)

1. Open the folder in VS Code with PlatformIO.
2. `platformio.ini` already lists required libs:
   - `espressif/esp32-camera`
   - `arduino-libraries/Arduino_TensorFlowLite`
3. **First build will compile with a placeholder model** (no real inference).

Upload & open Serial Monitor at **115200 baud**.

---

## Enable Real TinyML Inference

1. In Arduino IDE, open:
   **File → Examples → Arduino_TensorFlowLite → person_detection**
2. Copy `person_detect_model_data.h` from that example into: src/person_detect_model_data.h
(Overwrite the placeholder in this repo.)
3. Build & upload again. You should now see logs like:
and the **flash LED** turns **ON** when a person is detected.

---

## How It Works

- Camera frames at **160×120 GRAYSCALE** → resized to **96×96** (TinyML input).
- **TFLite Micro** runs an int8 quantized CNN (**~220 KB** tensor arena).
- If `person_score > 0.6`, we set **GPIO4 HIGH**.

Edit thresholds or actions to trigger **MQTT/HTTP**, relays, etc.

---

## Troubleshooting

- **Camera init failed** → Use a stable 5V supply; reseat the OV2640; verify FTDI wiring.
- **AllocateTensors failed** → Reduce `kTensorArenaSize` or ensure PSRAM is enabled (the AI Thinker has PSRAM).
- **No inference, only “Replace model header…”** → You’re still using the placeholder model. Replace the header as described.
- **Dim LED** → The on-board flash LED is GPIO4; it’s a tiny SMD LED. Use `digitalWrite(4, HIGH)` to verify.

---

## Next Steps

- Swap in your own **Edge Impulse** vision model (export Arduino library, integrate similar to this sketch).
- Add a lightweight **MJPEG stream** (use ESP32-CAM stream example) and overlay detection events.
- Record a short demo video and embed it in your blog.

---

## License

Starter code is MIT. TensorFlow Lite Micro model and example files are under their respective licenses.




