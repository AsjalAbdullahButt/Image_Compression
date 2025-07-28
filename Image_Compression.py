import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.fftpack import dct
from collections import Counter
import heapq
import os

# --- JPEG Compression Parameters ---
Q_Y = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

Q_C = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)

# --- Huffman Coding Classes and Functions ---


class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_table):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_table.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]


def generate_huffman_codes(root):
    codes = {}

    def traverse(node, prefix=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = prefix
        traverse(node.left, prefix + "0")
        traverse(node.right, prefix + "1")

    traverse(root)
    return codes


def huffman_encode_blocks(blocks):
    symbols = [pair for block in blocks for pair in block]
    freq = Counter(symbols)
    tree = build_huffman_tree(freq)
    codes = generate_huffman_codes(tree)
    bitstream = "".join(codes[symbol] for symbol in symbols)
    return bitstream, codes


# --- JPEG Compression Functions ---


def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    Y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    Cb = -0.1687 * img[:, :, 0] - 0.3313 * img[:, :, 1] + 0.5 * img[:, :, 2] + 128
    Cr = 0.5 * img[:, :, 0] - 0.4187 * img[:, :, 1] - 0.0813 * img[:, :, 2] + 128
    return Y, Cb, Cr


def zigzag_indices(n=8):
    return sorted(
        ((x, y) for x in range(n) for y in range(n)),
        key=lambda s: (s[0] + s[1], -s[0] if (s[0] + s[1]) % 2 else s[0]),
    )


def zigzag(block):
    indices = zigzag_indices()
    return np.array([block[i, j] for i, j in indices])


def run_length_encode(arr):
    result = []
    zero_count = 0
    for val in arr:
        if val == 0:
            zero_count += 1
        else:
            result.append((zero_count, val))
            zero_count = 0
    result.append((0, 0))  # End-of-block
    return result


def block_dct(img, Q):
    h, w = img.shape
    h_padded = h + (8 - h % 8) % 8
    w_padded = w + (8 - w % 8) % 8
    padded = np.zeros((h_padded, w_padded))
    padded[:h, :w] = img - 128
    blocks = []
    for i in range(0, h_padded, 8):
        for j in range(0, w_padded, 8):
            block = padded[i : i + 8, j : j + 8]
            dct_block = dct(dct(block.T, norm="ortho").T, norm="ortho")
            quant_block = np.round(dct_block / Q)
            zigzagged = zigzag(quant_block)
            rle = run_length_encode(zigzagged)
            blocks.append(rle)
    return blocks


def jpeg_compress_channel(channel, Q):
    blocks = block_dct(channel, Q)
    bitstream, huff_table = huffman_encode_blocks(blocks)
    return {"blocks": blocks, "bitstream": bitstream, "huffman_table": huff_table}


def jpeg_compress(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        return {
            "mode": "grayscale",
            "Y": jpeg_compress_channel(img, Q_Y),
            "shape": img.shape,
        }
    else:
        Y, Cb, Cr = rgb_to_ycbcr(img)
        return {
            "mode": "color",
            "Y": jpeg_compress_channel(Y, Q_Y),
            "Cb": jpeg_compress_channel(Cb, Q_C),
            "Cr": jpeg_compress_channel(Cr, Q_C),
            "shapes": {"Y": Y.shape, "Cb": Cb.shape, "Cr": Cr.shape},
        }


# --- GUI ---


class JPEGCompressorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JPEG Coder GUI (Compression + Huffman)")
        self.img_label = tk.Label(root, text="No image loaded", font=("Arial", 12))
        self.img_label.pack(pady=5)
        self.canvas = tk.Canvas(root, width=400, height=300, bg="gray")
        self.canvas.pack()
        self.stats_label = tk.Label(root, text="", font=("Arial", 10), justify="left")
        self.stats_label.pack(pady=10)
        self.load_btn = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_btn.pack(pady=5)
        self.compress_btn = tk.Button(
            root, text="Compress JPEG", command=self.compress_image
        )
        self.compress_btn.pack(pady=5)
        self.save_btn = tk.Button(
            root, text="Save Compressed Image", command=self.save_image
        )
        self.save_btn.pack(pady=5)
        self.image = None
        self.photo = None
        self.compressed = None

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        img_cv = cv2.imread(path)
        if img_cv is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        self.img_label.config(text=os.path.basename(path))
        self.image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(self.image).resize(
            (400, 300), Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.stats_label.config(text="Image loaded. Ready to compress.")

    def compress_image(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        self.compressed = jpeg_compress(self.image)
        text = f"Compression Mode: {self.compressed['mode'].upper()}\n"
        orig_size = self.image.nbytes
        compressed_size = len(self.compressed["Y"]["bitstream"]) // 8
        if self.compressed["mode"] == "grayscale":
            text += f"Image Shape: {self.compressed['shape']}\n"
            text += f"Y Blocks: {len(self.compressed['Y']['blocks'])}\n"
        else:
            shapes = self.compressed["shapes"]
            text += f"Y Shape: {shapes['Y']}, Blocks: {len(self.compressed['Y']['blocks'])}\n"
            text += f"Cb Shape: {shapes['Cb']}, Blocks: {len(self.compressed['Cb']['blocks'])}\n"
            text += f"Cr Shape: {shapes['Cr']}, Blocks: {len(self.compressed['Cr']['blocks'])}\n"
        text += f"\nOriginal Size: {orig_size} bytes\n"
        text += f"Compressed (Y stream) size: {compressed_size} bytes"
        self.stats_label.config(text=text)
        messagebox.showinfo("Success", "JPEG Compression Done!")

    def save_image(self):
        if self.compressed is None:
            messagebox.showwarning("No Compression", "Please compress an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg *.jpeg")]
        )
        if not path:
            return
        img_pil = Image.fromarray(self.image)
        img_pil.save(path)
        messagebox.showinfo(
            "Saved", f"Original image saved (not decompressed version)."
        )


# --- Run GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = JPEGCompressorGUI(root)
    root.mainloop()
