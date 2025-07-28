# JPEG Image Compression with Huffman Coding (Python GUI)

This project implements a simplified version of JPEG image compression using Python. It features:
- DCT-based transformation
- Quantization
- Zigzag scanning
- Run-Length Encoding (RLE)
- Huffman Coding
- A GUI built with Tkinter for easy interaction

## 📷 Features

- Load images (`.jpg`, `.jpeg`, `.png`, `.bmp`)
- Perform JPEG-like compression (grayscale and color)
- Huffman encoding on DCT coefficients
- View compression statistics
- Save the original image (compression metadata not embedded)

## 💡 How It Works

### 1. **Color Space Conversion**
If the image is in color (RGB), it is converted to YCbCr, separating luminance (Y) and chrominance (Cb, Cr) channels.

### 2. **Block Splitting and DCT**
Each channel is split into 8×8 blocks, and a 2D Discrete Cosine Transform (DCT) is applied to each block.

### 3. **Quantization**
Standard JPEG quantization matrices (`Q_Y` for luminance and `Q_C` for chrominance) are applied to reduce frequency detail.

### 4. **Zigzag Scan**
The quantized block is converted to a 1D array using zigzag ordering for efficient RLE.

### 5. **Run-Length Encoding (RLE)**
Consecutive zeros in the zigzag array are replaced with `(zero_count, value)` pairs.

### 6. **Huffman Encoding**
A Huffman tree is built for all RLE pairs, and the data is encoded into a binary bitstream.

## 🧪 Example

Upon compressing an image:
- You see the number of 8x8 blocks per channel
- Original image size vs. compressed Y-channel bitstream size (approximate)
- The actual image is not altered, only the stats are shown

## 🖼 GUI Overview

| Button                 | Description                                        |
|------------------------|----------------------------------------------------|
| `Load Image`           | Opens a dialog to select and load an image         |
| `Compress JPEG`        | Performs JPEG compression with Huffman coding      |
| `Save Compressed Image`| Saves the original image (not decompressed)        |

## 🧰 Requirements

- Python 3.7+
- Required libraries:
  ```bash
  pip install numpy pillow opencv-python scipy
