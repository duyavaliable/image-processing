# Ứng Dụng Xử Lý Ảnh - Computer Vision

Ứng dụng web xử lý ảnh toàn diện với giao diện thân thiện, hỗ trợ nhiều phương pháp biến đổi ảnh, khử nhiễu, và nhận diện đối tượng.

## 📋 Tổng Quan

Ứng dụng cho phép người dùng tải lên ảnh và áp dụng các kỹ thuật xử lý ảnh khác nhau thông qua giao diện kéo-thả trực quan. Backend xử lý bằng Flask + OpenCV, frontend sử dụng vanilla JavaScript.

## 🚀 Cài Đặt & Chạy

### Yêu Cầu
- Python 3.8+
- pip

### Các Bước

1. **Clone repository và di chuyển vào thư mục:**
   ```bash
   cd computer-vision-mid-term
   ```

2. **Tạo môi trường ảo:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Cài đặt dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Tải mô hình YOLO (cho nhận diện đối tượng):**
   - Tải [YOLOv5s ONNX](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx)
   - Đặt vào thư mục `models/yolov5s.onnx`

5. **Chạy server:**
   ```bash
   python server.py
   ```

6. **Mở trình duyệt:**
   ```
   http://localhost:5500
   ```

## 📁 Cấu Trúc Dự Án

```
.
├── index.html              # Giao diện chính
├── server.py               # Flask backend
├── requirements.txt        # Python dependencies
├── src/
│   ├── app.js             # Logic frontend
│   └── styles.css         # Giao diện CSS
└── models/
    └── yolov5s.onnx       # Mô hình YOLO (tải riêng)
```

## 🎨 Các Chức Năng

### 1. Biến Đổi Cường Độ (Intensity Transformations)

#### **Grayscale 8-bit**
- Chuyển ảnh màu sang ảnh xám
- Công thức: `Gray = 0.299×R + 0.587×G + 0.114×B`
- **Ứng dụng:** Tiền xử lý cho các thuật toán phân tích ảnh

#### **Negative Images**
- Đảo ngược giá trị pixel: `output = 255 - input`
- **Ứng dụng:** Tăng cường chi tiết vùng tối, chụp X-quang y tế

#### **Logarithmic Transformation**
- Công thức: `s = c × log(1 + r)`
- **Ứng dụng:** Mở rộng vùng tối, nén vùng sáng (ảnh chụp phơi sáng kém)
- **Tham số:** `c` điều chỉnh độ mạnh (mặc định: 2.0)

#### **Power-Law (Gamma Correction)**
- Công thức: `s = r^γ`
- **Tham số:**
  - `γ < 1`: Làm sáng ảnh
  - `γ > 1`: Làm tối ảnh
  - `γ = 1`: Không thay đổi
- **Ứng dụng:** Điều chỉnh độ sáng màn hình, sửa gamma camera
- **Điều khiển:** Nhập giá trị gamma thủ công (0.1 - 10)

#### **Piecewise Linear (Contrast Stretch)**
- Ánh xạ phi tuyến theo 2 điểm kiểm soát `(r1,s1)` và `(r2,s2)`
- **Ứng dụng:** Tăng tương phản vùng quan tâm
- **Tham số:**
  - `r1, r2`: Ngưỡng input (1-255)
  - `s1, s2`: Giá trị output tương ứng (0-255)

#### **Thresholding (Binary)**
- Phân ngưỡng ảnh: `output = 255 nếu input > T, ngược lại 0`
- **Tham số:** Ngưỡng `T` (0-255, mặc định 128)
- **Ứng dụng:** Phân đoạn đối tượng, nhị phân hóa ảnh

### 2. Xử Lý Nhiễu (Denoising)

#### **Phương pháp khử nhiễu thủ công (Manual Denoise)**

Cho phép người dùng chọn phương pháp phù hợp với loại nhiễu:

##### **Median Filter**
- **Nhiễu phù hợp:** Salt-and-Pepper (nhiễu muối tiêu)
- **Tham số:** Kernel size (3-31, lẻ)
- **Ưu điểm:** Bảo toàn biên, loại bỏ hiệu quả nhiễu xung
- **Ứng dụng:** Ảnh bị lỗi truyền, nhiễu điểm ngẫu nhiên

##### **Bilateral Filter**
- **Nhiễu phù hợp:** Gaussian (nhiễu cộng)
- **Tham số:** 
  - `d`: Đường kính vùng lọc (kernel size)
  - `sigma`: Tự động = d × 10
- **Ưu điểm:** Làm mượt vùng đồng nhất, giữ biên sắc nét
- **Ứng dụng:** Nhiễu cảm biến, nhiễu nhiệt

##### **Non-local Means (NLM)**
- **Nhiễu phù hợp:** Gaussian, nhiễu texture
- **Tham số:** `h` (cường độ lọc) = kernel × 2
- **Ưu điểm:** Bảo toàn texture, chi tiết cao
- **Ứng dụng:** Ảnh có nhiều texture lặp lại

##### **Lee Filter** *(Server-side implementation)*
- **Nhiễu phù hợp:** Speckle (nhiễu nhân)
- **Tham số:** Window size (3-31)
- **Công thức:** Kết hợp trung bình cục bộ và variance
- **Ứng dụng:** Ảnh SAR, radar, siêu âm

#### **Phân Tích Histogram**

Server tự động phân tích histogram và gợi ý phương pháp:

- **`extreme_ratio`**: Tỉ lệ pixel đen/trắng thuần (`0` hoặc `255`)
  - `> 2%` → Gợi ý Median (Salt-and-Pepper)

- **`variance`**: Độ phân tán pixel
  - `100 < variance < 2000` → Gợi ý Bilateral (Gaussian)

- **`cv` (Coefficient of Variation)**: `std / mean`
  - `> 0.28` → Gợi ý Lee (Speckle)

- **`suggested_method`**: Phương pháp được gợi ý dựa trên phân tích

**Lưu ý:** Histogram params được trả về từ server nhưng **không hiển thị trên UI** để giao diện gọn gàng.

### 3. Tăng Cường Ảnh (Enhancement)

#### **Histogram Equalization**
- Cân bằng histogram để tăng tương phản toàn cục
- **Ứng dụng:** Ảnh có độ tương phản thấp, phân bố cường độ không đều

#### **Sharpening (Laplacian)**
- Tăng cường biên bằng bộ lọc Laplacian
- **Công thức:** `output = input - ∇²input`
- **Hỗ trợ:** Ảnh màu (xử lý từng kênh) và ảnh xám
- **Ứng dụng:** Làm rõ biên, chi tiết mờ

### 4. Lọc Tần Số (Frequency Domain Filtering)

Áp dụng DFT/FFT để lọc trong miền tần số:

#### **Lowpass Filters (Lọc thông thấp)**
- Loại bỏ tần số cao (nhiễu, chi tiết nhỏ)
- **Loại:**
  - **Ideal:** Cắt cứng tại tần số `D0`
  - **Butterworth:** Chuyển tiếp mượt (order `n`)
  - **Gaussian:** Chuyển tiếp mượt nhất

#### **Highpass Filters (Lọc thông cao)**
- Tăng cường biên, loại bỏ background
- **Loại:** Tương tự lowpass

**Tham số:**
- `Cutoff (D0)`: Tần số cắt (5-200, mặc định 30)
- `Order (n)`: Bậc Butterworth (1-10, mặc định 2)

**Ứng dụng:**
- Lowpass: Làm mượt, giảm nhiễu
- Highpass: Phát hiện biên, tăng cường chi tiết

### 5. Khôi Phục Ảnh (Restoration)

#### **Wiener Filter** *(Approximation)*
- Lọc tối ưu kết hợp thông tin ảnh và nhiễu
- **Implementation:** Sử dụng bilateral filter làm xấp xỉ
- **Tham số:** `noise_var` (variance của nhiễu, mặc định 100)
- **Ứng dụng:** Deblur, khử nhiễu khi biết PSF

### 6. Nhận Diện Đối Tượng (Object Detection)

#### **YOLO (You Only Look Once)**
- Mô hình: YOLOv5s ONNX
- **Tham số:**
  - `conf_thresh`: Ngưỡng confidence (mặc định 0.25)
  - `iou_thresh`: Ngưỡng IoU cho NMS (mặc định 0.45)
  - `inp_size`: Kích thước input (mặc định 640)

#### **Các Lớp Hỗ Trợ (80 lớp COCO)**
- Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat...
- Dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe...
- Bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich...
- Chair, couch, bed, dining table, toilet, tv, laptop, mouse, keyboard...
- *(Xem full danh sách 80 lớp trong [`server.py`](server.py))*

#### **Output**
- **Hình ảnh:** Bounding boxes + labels + confidence scores
- **Message:** `Detected {N} objects`
- **Params (không hiển thị trên UI):**
  ```json
  {
    "detected_objects": [
      {"label": "person", "confidence": 0.92, "bbox": [x, y, w, h]},
      ...
    ],
    "total_detected": N,
    "backend": "yolo_detect"
  }
  ```

## 🎮 Cách Sử Dụng

### Giao Diện Chính

1. **Tải ảnh lên:**
   - Kéo & thả ảnh vào vùng "drop area"
   - Hoặc click **"Tải ảnh lên"** để chọn file

2. **Chọn chế độ xử lý:**
   - Dropdown menu hiển thị tất cả các chế độ
   - Các điều khiển tương ứng sẽ hiện/ẩn tự động

3. **Điều chỉnh tham số:**
   - Gamma: Nhập giá trị → Enter để áp dụng
   - Piecewise: Nhập r1, s1, r2, s2 → Enter
   - Denoise: Chọn method + kernel size
   - Sharpening: Chỉ có Laplacian (tự động)
   - Frequency Filter: Chọn type + cutoff + order
   - Threshold: Nhập T → Enter

4. **Xem kết quả:**
   - Ảnh preview hiển thị ngay lập tức
   - Message phía dưới mô tả kết quả xử lý

### Keyboard Shortcuts

- **Enter** trong các input field:
  - Gamma, Piecewise params: Áp dụng ngay
  - Denoise kernel: Áp dụng ngay
  - Frequency filter params: Áp dụng ngay
  - Threshold T: Áp dụng ngay

- **Debounce** cho Denoise kernel input: 800ms (tự động re-process khi ngừng nhập)

### Format Ảnh Hỗ Trợ

- **JPEG/JPG**: Full support (server decode)
- **PNG**: Full support
- **TIFF**: Server-side processing only (browser không render TIFF trực tiếp)
- **Max size:** 8MB

## 🔧 Chi Tiết Kỹ Thuật

### Backend (Flask + OpenCV)

#### **Robust Image Decoding**
- [`ensure_gray_uint8_from_bytes()`](server.py): Decode ảnh từ bytes
- Hỗ trợ: RGBA, 16-bit, float, TIFF
- Fallback qua nhiều `cv2.IMREAD_*` flags

#### **Noise Detection Heuristics**
- [`detect_noise_type()`](server.py): Phân tích loại nhiễu
- **Primary:** Frequency domain (FFT peaks, energy ratio)
- **Fallback:** Histogram features (extreme_ratio, variance, CV)

#### **Adaptive Filtering**
- [`adaptive_median_filter()`](server.py): Tăng window size động
- [`lee_filter()`](server.py): Speckle noise reduction

#### **Frequency Domain Processing**
- [`notch_filter_auto()`](server.py): Loại bỏ periodic noise
- Percentile clipping (p01-p99) để tránh outliers

### Frontend (Vanilla JavaScript)

#### **Event-Driven Architecture**
- Mode change → `updateControlsVisibility()` → re-process
- Input events → debounced re-processing
- File drop → `handleFile()` → upload → display

#### **Client-Side Processing** *(optional fallback)*
- [`toGray8DataURL()`](src/app.js): Grayscale conversion
- [`toLogDataURL()`](src/app.js): Log transform
- [`toMedianDataURL()`](src/app.js): Median filter (slow, demo only)

**Server-first Strategy:**
- Hầu hết processing trên server (OpenCV mạnh hơn)
- Client chỉ xử lý preview nhanh (nếu ảnh nhỏ)

#### **Preview Management**
- [`showInputPreview()`](src/app.js): Preview ảnh gốc
  - TIFF: Upload server để convert → preview
  - Other: Object URL hoặc fallback server
- [`showPreview()`](src/app.js): Preview ảnh kết quả

## 🐛 Debug & Logging

### Server Logs
```python
print(f"[mode] processing image: {mode}, extras: {extras}")
```

### Client Logs
```javascript
console.log('[handleFile] mode:', mode, 'file:', file.name);
console.debug('[uploadFileToServer] response:', resp);
```

### Common Issues

#### **1. TIFF không hiển thị preview**
- ✅ Server xử lý TIFF → convert sang PNG base64
- ⚠️ Browser không render TIFF trực tiếp

#### **2. Params không hiển thị cho `denoise_manual` / `count_objects`**
- ✅ Intentional: UI chỉ hiển thị message ngắn gọn
- 🔍 Xem params trong console: `console.debug('[handleFile] server response:', resp)`

#### **3. JPEG decode error**
- ✅ Fallback: [`ensure_gray_uint8_from_bytes()`](server.py) thử nhiều flags
- ⚠️ Ảnh JPEG corrupt/metadata phức tạp có thể fail

#### **4. Frequency filter output tối/sáng bất thường**
- ✅ Normalize: `(img - min) / (max - min) × 255`
- ⚠️ Kiểm tra `cutoff` và `order` có hợp lý không

## 📚 Tài Liệu Tham Khảo

### Lý Thuyết
- **Gonzalez & Woods:** *Digital Image Processing* (3rd Ed.)
  - Chapter 3: Intensity Transformations
  - Chapter 5: Image Restoration
  - Chapter 4: Filtering in Frequency Domain

### Thuật Toán
- **Median Filter:** Tukey, 1977
- **Bilateral Filter:** Tomasi & Manduchi, 1998
- **Non-local Means:** Buades et al., 2005
- **Lee Filter:** Lee, 1980 (for speckle noise)
- **YOLOv5:** Ultralytics, 2020

### OpenCV Documentation
- [cv2.medianBlur](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)
- [cv2.bilateralFilter](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430689ed)
- [cv2.fastNlMeansDenoising](https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93)
- [cv2.dft / cv2.idft](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gadd6cf9baf2b8b704a11b5f04aaf4f39d)

## 🤝 Đóng Góp

Nếu bạn muốn thêm tính năng:
1. Fork repo
2. Tạo feature branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m 'Add some AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Tạo Pull Request

## 📝 License

MIT License - Xem [LICENSE](LICENSE) để biết chi tiết.

## 👨‍💻 Tác Giả

**[Tên của bạn]**
- Email: [email@example.com]
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Note:** README này được tạo dựa trên cấu trúc thực tế của dự án. Mọi tính năng đã được kiểm tra và hoạt động ổn định với Python 3.8+ và các dependencies trong [`requirements.txt`](requirements.txt).