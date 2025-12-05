# Báo cáo — Các phương pháp xử lý ảnh dùng trong dự án

Tài liệu tóm tắt những lý thuyết và luồng xử lý đã triển khai trong project. Mục đích: giải thích các phương pháp, tham số và cách hoạt động chung (không nêu chi tiết file/đoạn code).

## 1. Tổng quan
Ứng dụng nhận ảnh đầu vào, phân tích loại nhiễu, lựa chọn pipeline phù hợp, thực hiện khử nhiễu / biến đổi ảnh và trả về ảnh đã xử lý cùng các tham số chẩn đoán.

## 2. Mô hình nhiễu (Noise Models)
- Gaussian (additive): nhiễu theo phân bố chuẩn (sensor, thermal). Đặc trưng: tăng biến thiên đều trên ảnh.  
- Salt-and-Pepper: pixel bị bật về 0 hoặc 255 (lỗi truyền/sensor). Đặc trưng: spike ở hai đầu histogram.  
- Uniform: phân bố đều trên khoảng [a, b].  
- Periodic (tuần hoàn): nhiễu tạo vân lặp; xuất hiện như các đỉnh (peaks) trong miền tần số.  
- Speckle (multiplicative): nhiễu nhân, phổ biến trong radar/siêu âm; đặc trưng bởi hệ số biến động cao.

## 3. Phân tích & nhận diện nhiễu (heuristics)
Dùng nhiều đặc trưng để phân loại:
- extreme_ratio = tỉ lệ pixel =0 hoặc =255 → nghi salt‑and‑pepper.
- variance / MAD (median absolute deviation) → kiểm tra Gaussian.
- CV = std/mean → nghi speckle nếu CV cao.
- DFT/FFT magnitude + peak detection + peak_energy_ratio → periodic nếu có nhiều đỉnh có năng lượng tương đối lớn.
Kết hợp các tính năng (rule‑based) giảm sai phân loại; ngưỡng cần điều chỉnh theo dataset.

## 4. Lọc miền không gian (Spatial Filtering)
- Mean (averaging): làm mượt, giảm nhiễu tần số cao; làm mờ biên.  
- Median: bảo toàn biên, rất hiệu quả với salt‑and‑pepper.  
- Max/Min, Midpoint, Geometric/Harmonic/Contra‑harmonic: các biến thể cho các loại nhiễu đặc thù.

## 5. Lọc thích nghi (Adaptive Filtering)
- Adaptive mean: điều chỉnh theo nhiễu cục bộ.  
- Adaptive median: tăng cửa sổ khi cần, xử lý salt‑and‑pepper nặng.

## 6. Miền tần số (Frequency Domain)
- FFT/DFT để phân tích periodic noise.  
- Notch/Bandreject: loại bỏ tần số xác định (dùng để xử lý periodic).  
- Low‑pass / High‑pass / Butterworth / Gaussian filters: giảm/tăng thành phần tần số tương ứng.  
Lưu ý: sau inverse DFT cần chuẩn hóa/clipping (percentile clip) để tránh outlier ảnh hưởng hiển thị.

## 7. Wiener filtering & Deblurring
- Wiener filter tối ưu theo SNR, kết hợp thông tin ảnh và nhiễu; phù hợp khi biết hoặc ước lượng PSF.  
- Inverse filtering và deconvolution dùng khi PSF biết, nhưng rất nhạy với nhiễu — thường kết hợp Wiener.

## 8. Xử lý hình thái & phân đoạn
- Erosion / Dilation / Opening / Closing: xử lý cấu trúc, loại nhiễu nhỏ, lấp lỗ.  
- Connected Component Labeling (CCL): đếm/tách/phân vùng đối tượng, vẽ label hoặc xuất thống kê.

## 9. Biến đổi cường độ & tiền xử lý
- Grayscale 8‑bit, Negative, Histogram equalization, Contrast stretch.  
- Thresholding: fixed threshold / Otsu.  
- Log transform: tăng chi tiết vùng tối.  
- Power‑law (Gamma correction): sửa gamma (auto hoặc bằng tay).  
- Piecewise linear: điều chỉnh tương phản theo điểm kiểm soát.

## 10. Luồng gợi ý "Denoise (Auto)"
1. Decode ảnh (giữ màu nếu cần). Tính luminance để phân tích.  
2. Tính features: extreme_ratio, variance, MAD, CV, FFT magnitude peaks, peak_energy_ratio.  
3. Quyết định backend:
   - Salt‑and‑pepper → Adaptive median + morphological opening.
   - Gaussian → Bilateral / Non‑local Means / Wiener (nếu PSF biết).
   - Periodic → Notch filter (frequency domain), xử lý theo kênh màu nếu cần.
   - Speckle → Lee / Geometric mean.
4. Áp dụng filter (nếu ảnh màu, xử lý từng kênh và ghép lại).  
5. Post‑processing: morphological cleaning, percentile clipping (p01–p99) và scale lại, optional contrast stretch.  
6. Trả về ảnh (PNG base64) + tham số chẩn đoán (noise_type, phương pháp, các giá trị thống kê).

## 11. Những giới hạn & lưu ý vận hành
- Ngưỡng tĩnh dễ gây sai; nên hiệu chỉnh theo tập dữ liệu hoặc kết hợp nhiều đặc trưng.  
- Periodic detection cần đo tỉ lệ năng lượng của đỉnh so với năng lượng tổng (peak_energy_ratio) để tránh false positive từ texture.  
- Notch filter quá mạnh sẽ làm mất chi tiết → giới hạn số peak / bán kính notch và dùng hậu xử lý normalize.  
- Để giữ màu, áp dụng lọc theo kênh màu (BGR/HSV) thay vì convert toàn bộ sang grayscale.

## 12. Gợi ý test & ví dụ lệnh
- Denoise auto:
```sh
curl -F "file=@/path/to/img.jpg" -F "mode=denoise" http://localhost:5500/api/process
```
- Count objects:
```sh
curl -F "file=@/path/to/img.jpg" -F "mode=count_objects" -F "target_class=person" http://localhost:5500/api/process
```

## 13. Ghi chú khi debug
- Luôn log các giá trị raw (extreme_ratio, num_peaks, peak_energy_ratio, variance, MAD, CV) để tinh chỉnh thresholds.  
- Trả về JSON-safe types (convert numpy → native) để tránh lỗi serializing.  
- Khi ảnh output không như mong muốn: kiểm tra dtype, range trước khi encode (phải là uint8 và nằm trong [0,255]).

## 14. Tài liệu tham khảo ngắn
- Gonzalez & Woods, Digital Image Processing — chương về noise, filtering, frequency domain.  
- Buades et al., Non‑local Means; Wiener filter; Lee filter cho speckle.

---

Nếu muốn, tôi có thể bổ sung ví dụ hình ảnh trước/sau cho từng phương pháp hoặc liệt kê lệnh curl chi tiết hơn.