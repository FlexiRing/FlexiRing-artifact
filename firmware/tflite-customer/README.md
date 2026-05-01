# tflite-customer

This repository contains embedded firmware projects based on the Ambiq Apollo3 platform, including a gesture recognition example built with TensorFlow Lite Micro.

## 1. Project Scope

- Target device: FlexiRing (in-house wearable prototype)
- Main MCU: Ambiq Apollo3 Blue
- Primary languages: C / C++
- Reference build entry: `boards/apollo3_evb/examples/hello_fault/keil6`

Notes:

- The `apollo3_evb` project tree is mainly provided as a reference entry for Keil-based build and debug.
- The core gesture inference logic has been executed and validated on FlexiRing (Apollo3 Blue).
- This repository provides an inference demo program that can run with the accompanying input data for verification.
- This is not the final production firmware and does not include the full end-to-end system stack (for example, complete task scheduling, full sensing pipeline, communication stack, and product-level integration).

## 2. Directory Overview

- `boards/`: Board support and example projects
- `mcu/`: MCU-related HAL and low-level implementation
- `devices/`: Peripheral drivers
- `utils/`: Utility libraries
- `ambiq_ble/`: BLE stack and application examples
- `third_party/`: Third-party libraries (for example, FreeRTOS, mbedtls)
- `docs/`: Project documentation
- `tools/`: Build, flashing, and helper tools

Directories directly related to gesture inference:

- `boards/apollo3_evb/examples/hello_fault/src`
- `boards/apollo3_evb/examples/hello_fault/keil6`

## 3. Requirements

- Hardware:
  - FlexiRing (Apollo3 Blue) or Apollo3 EVB (for reference build validation)
- Development environment:
  - Keil MDK (MDK-ARM v5 with ARMClang recommended)
- Required packs/components:
  - `AmbiqMicro.Apollo_DFP.1.5.4`
  - CMSIS / Machine Learning components referenced by the project (Keil will prompt if missing)
- Flashing/debug:
  - J-Link (or equivalent flashing method supported by Keil)
- Serial tools:
  - Any serial terminal (for example, SSCOM, PuTTY, MobaXterm, pyserial)

## 4. Build and Flash

1. Open project:

   `boards/apollo3_evb/examples/hello_fault/keil6/hello_fault.uvprojx`

2. In Keil, select target: `hello_fault`
3. Build the project
4. Connect hardware and flash firmware (Download / Load)

If Keil reports missing components on first open, install them via Keil Pack Installer before building.

## 5. Runtime Flow (Gesture Inference)

The program does not run continuous inference automatically after boot. A single inference is triggered via serial command.

Scope clarification:

- This demo is intended to demonstrate and validate the gesture inference path.
- Input is provided externally in a predefined format; output is inference status plus class scores.
- This flow is for algorithm reproduction and verification and does not represent the final full firmware workflow on FlexiRing.

### 5.1 Serial Settings

- Baud rate: `115200`
- Data bits: `8`
- Parity: `None`
- Stop bits: `1`
- Flow control: `None`

### 5.2 Trigger

Send a single character to the device over serial:

- `S`

After receiving `S`, the program starts receiving input data and performs one inference pass.

### 5.3 Input Data Format

- Input length: `180 x 6 = 1080` float values
- Axis order: `ax, ay, az, gx, gy, gz`
- Delimiters: commas and newlines
- First line: treated as header and skipped (recommended to keep)

CSV example (format only, not a full 1080-value sample):

```csv
ax,ay,az,gx,gy,gz
0.01,0.02,0.98,1.20,-0.30,0.40
0.02,0.01,0.97,1.18,-0.28,0.41
...
```

## 6. Output

After one inference pass, serial output includes:

1. Inference status code (`0` usually indicates success)
2. Per-class scores (one line per class)

Example output (illustrative):

```text
0
0 0.012345
1 0.103210
2 0.000421
...
```

## 7. FAQ

### Q1: The project opens but fails to build

- Check whether your Keil version supports ARMClang
- Check whether all required packs/components are installed

### Q2: Flash succeeds but no inference result is printed

- Check serial settings (`115200 8N1`)
- Check whether trigger character `S` is sent
- Check whether CSV includes a header and follows the required format

### Q3: Results are abnormal or inference fails

- Check whether input contains all 1080 float values
- Check whether serial transmission is truncated or contaminated by extra characters

## 8. Reproducibility Notes (For Submission Materials)

To make peer review reproduction easier, provide:

- Firmware version or commit hash
- Device information (FlexiRing hardware revision, Apollo3 Blue variant)
- One complete replayable CSV sample (1080 values)
- One matched serial log snippet (input + output)
