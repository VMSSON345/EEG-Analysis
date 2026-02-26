#  EEG Based Event Classification using Machine Learning

Dự án này tập trung vào việc phân loại các sự kiện dựa trên tín hiệu điện não đồ (EEG) thông qua các mô hình Học máy (Machine Learning).  
Dự án được thực hiện tại **Viện Trí tuệ Nhân tạo, Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội**.

---

##  Giới thiệu Dự án

Mục tiêu chính của dự án là phân loại các chuỗi tín hiệu EEG thành hai loại sự kiện:

- **Standard** (Tiêu chuẩn - nhãn 0)  
- **Target** (Mục tiêu - nhãn 1)

Trong đó, các sự kiện **Target** sẽ tạo ra một phản ứng não đặc trưng gọi là **sóng P300** (xuất hiện khoảng 300ms sau kích thích), trong khi các sự kiện **Standard** thì không.

Để giải quyết bài toán này, chúng tôi đã xây dựng một pipeline hoàn chỉnh bao gồm:

- Tiền xử lý tín hiệu EEG  
- Trích xuất đặc trưng  
- Huấn luyện mô hình học máy  

---

## Tập Dữ liệu (Dataset)

- **Nguồn:** Trích xuất từ tập *Nencki-Symfonia EEG/ERP Dataset* trên GigaScience Database (GigaDB).  
- **Quy mô:** Dữ liệu thô ban đầu gồm 127 kênh, sau khi loại bỏ nhiễu và kênh EOG, còn lại 105 kênh EEG sạch.  
- **Đối tượng:** Tổng cộng có 42 đối tượng tham gia, dữ liệu cuối cùng sau khi lọc đối tượng kém chất lượng bao gồm 38 subjects.

---

## Quy trình Xử lý (Pipeline)

###  Tiền xử lý Tín hiệu (Signal Preprocessing)

- **Loại bỏ kênh:** Xóa 2 kênh tham chiếu (TP9, TP10) và 20 kênh viền do chứa nhiều nhiễu cơ/mắt và không mang thông tin vỏ não trung tâm.  
- **Tham chiếu chung:** Dán nhãn kênh EOG (AFp1, AFp2) và áp dụng phương pháp **Common Average Referencing (CAR)** để loại bỏ nhiễu toàn cục.  
- **Lọc tần số:**  
  - Notch filter 50 Hz để loại bỏ nhiễu dòng điện  
  - Band-pass filter 0.1–40 Hz để giữ lại sóng P300  
- **Loại bỏ nhiễu nâng cao:** Chạy **ICA** và sử dụng mô hình deep learning **ICLabel** để tự động loại bỏ nhiễu sinh học (mắt, cơ, tim).  
- **Epoching & Baseline Correction:** Phân mảnh tín hiệu từ -200 ms đến +800 ms và hiệu chỉnh đường cơ sở dựa trên khoảng pre-stimulus.  
- **Lọc biên độ:** Loại bỏ các epoch có biên độ tuyệt đối vượt quá ±100 µV.

---

### Trích xuất Đặc trưng (Feature Extraction)

- **Miền tần số:**  
  - Tính toán PSD bằng phương pháp Welch  
  - Trích xuất 5 dải băng tần: Delta, Theta, Alpha, Beta, Gamma  
  - Biến đổi log10 để ổn định dữ liệu  

- **Miền thời gian:**  
  - Tập trung vào cửa sổ ERP từ 250 ms đến 600 ms  
  - Trích xuất:  
    - Biên độ trung bình  
    - Biên độ đỉnh  
    - Độ trễ đỉnh của sóng P300  

- **Đặc trưng chuỗi:**  
  - Sử dụng Hidden Markov Models (HMM) với 5 trạng thái ẩn  
  - Áp dụng trên các kênh trung tâm-đỉnh (Pz, Cz, Fz, P3, P4)  
  - Tính điểm log-likelihood  

---

###  Huấn luyện Mô hình

- **Tách dữ liệu:**  
  - Subject-based Splitting (kiểm tra khả năng tổng quát hóa trên người mới)  
  - Trial-based Splitting (chia ngẫu nhiên toàn bộ trial)  

- **Cân bằng dữ liệu:**  
  - Áp dụng thuật toán **SMOTE** trên tập huấn luyện để xử lý tình trạng lệch nhãn  

- **Mô hình sử dụng:**  
  - XGBoost  
  - Random Forest  
  - LightGBM  
  - MLPClassifier  
  - CatBoost  
  - ExtraTrees Classifier  

---

##  Kết quả Đánh giá 

- **Ưu thế của Ensemble Learning:**  
  Các mô hình dạng cây (XGBoost, Random Forest, LightGBM, CatBoost) hoạt động ổn định nhất, đạt độ chính xác lên tới **0.92**.

- **Mô hình tốt nhất:**  
  **XGBoost** đạt hiệu suất cao nhất nhờ khả năng nắm bắt các tương tác phi tuyến phức tạp.

- **Vai trò của HMM:**  
  Việc thêm HMM post-processing đặc biệt hữu ích cho MLPClassifier (giúp tăng độ chính xác đáng kể), trong khi các mô hình dạng cây ít bị ảnh hưởng hơn.

- **Độ bền bỉ:**  
  Khi thu hẹp dải tần số, các mô hình Tree-based vẫn giữ được độ chính xác ổn định, chứng minh tính bền bỉ của phương pháp Ensemble Learning.

---

##  Sinh viên & Giảng viên Hướng dẫn
###  Sinh viên thực hiện

- Nguyễn Bá Quang (23020412)  
- Vũ Minh Sơn (23020424)  
- Phạm Thế Trung (23020442)  
- Mai Minh Tùng (23020432)  
- Tạ Quang Linh (23020396)  
- Trần Doãn Thắng (23020438)  

###  Giảng viên hướng dẫn

- GS. TS. Nguyễn Linh Trung  
- GS. Guy Nagels  
- TS. Nguyễn Thế Hoàng Anh  

---

##  Ghi chú

Dự án phục vụ mục đích nghiên cứu học thuật trong lĩnh vực Brain-Computer Interface (BCI) và Machine Learning.
