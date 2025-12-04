# Báo cáo — Các phương pháp xử lý ảnh dùng trong dự án

Tài liệu tóm tắt những lý thuyết xử lý ảnh đã áp dụng trong dự án (chỉ nêu khái niệm & phương pháp, không liệt kê tên hàm hay file).

## 1. Tổng quan
Ứng dụng nhận ảnh đầu vào, phân tích loại nhiễu và áp dụng pipeline xử lý phù hợp để trả về ảnh đã khử nhiễu/biến đổi.

## 2. Mô hình nhiễu (Noise Models)
- Gaussian noise: nhiễu ngẫu nhiên theo phân bố chuẩn (do cảm biến, nhiệt).  
- Salt-and-pepper noise: pixel ngẫu nhiên thành 0 hoặc 255 (lỗi truyền dẫn/sensor).  
- Uniform noise: phân bố đều trên một khoảng.  
- Periodic noise: nhiễu tuần hoàn tạo vân lặp rõ rệt trong miền tần số.  
- Speckle (multiplicative): nhiễu nhân, thường gặp trong radar/siêu âm.

## 3. Lọc trong miền không gian (Spatial Filtering)
- Mean filter (averaging): làm mịn, giảm nhiễu Gaussian nhẹ nhưng làm mờ biên.  
- Median filter: tốt cho salt-and-pepper, bảo toàn biên.  
- Max/Min, Midpoint, Geometric/Harmonic/Contra-harmonic mean: các biến thể phù hợp cho các kiểu nhiễu cụ thể.

## 4. Lọc thích nghi (Adaptive Filtering)
- Adaptive mean: điều chỉnh độ mượt dựa trên ước lượng nhiễu cục bộ.  
- Adaptive median: mở rộng median với kích thước cửa sổ thay đổi, hiệu quả cho salt-and-pepper nặng.

## 5. Xử lý miền tần số (Frequency Domain Filtering)
- Phân tích DFT/FFT để phát hiện và phân biệt periodic noise.  
- Notch filter: loại tần số cụ thể gây periodic noise bằng cách đặt notch trong phổ.  
- Low-pass / High-pass / Bandreject để giảm hoặc tăng tần số theo mục tiêu.

## 6. Wiener filtering (Lọc tối ưu)
- Wiener filter tối ưu theo SNR, kết hợp thông tin thống kê của ảnh và nhiễu, dùng cho denoising và deblurring khi biết đặc tính PSF/ổ nhiễu.

## 7. Mô hình suy giảm ảnh (Image Degradation Model)
- Mô hình cơ bản: g = h * f + η (ảnh quan sát = ảnh gốc * PSF + nhiễu).  
- Dùng làm nền tảng cho các phương pháp khôi phục (inverse filtering, Wiener).

## 8. Xử lý hình thái (Morphological Operations)
- Erosion, Dilation, Opening, Closing: các phép toán hình thái học cơ bản để xử lý cấu trúc ảnh.  
- Ứng dụng trong loại bỏ nhiễu nhỏ, lấp đầy khoảng trống, và phân đoạn đối tượng.

## 9. Luồng xử lý "Denoise (Auto)" trong code
- Nhận diện loại nhiễu tự động và áp dụng bộ lọc phù hợp: adaptive median cho salt-and-pepper, bilateral/fastNlMeans cho Gaussian, notch filter cho periodic, và Lee filter cho speckle.

## 10. Những giới hạn & lưu ý đã gặp trong thực nghiệm
- Phát hiện periodic noise có thể bị nhầm lẫn với nhiễu Gaussian do ngưỡng nhạy.  
- Notch filtering có thể làm mất tương phản nếu không điều chỉnh lại sau khi lọc.  
- Ảnh màu cần xử lý từng kênh riêng biệt để bảo toàn thông tin màu sắc.

## 11. Cách chạy & test
- Cài đặt môi trường:
```sh
pip install -r requirements.txt
```
- Chạy server:
```sh
python server.py
```
Truy cập giao diện: http://localhost:5500 (hoặc host:port theo cấu hình).

---

Nếu muốn, tôi có thể:
- Chuyển toàn bộ nội dung này vào file README (áp dụng patch).  
- Thêm ví dụ test case và lệnh curl để test API.  
Trả lời: "Áp dụng README" nếu muốn tôi ghi nội dung vào README.MD.