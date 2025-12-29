# Dự Án Phân Cụm Khách Hàng Shop

## 📊 Tổng Quan

Dự án này triển khai pipeline phân cụm khách hàng toàn diện sử dụng kỹ thuật khai thác luật kết hợp và phân cụm. Cách tiếp cận biến đổi dữ liệu giao dịch thành insights marketing có thể hành động được bằng:

1. **Khai Thác Luật Kết Hợp**: Khám phá mối quan hệ sản phẩm sử dụng thuật toán FP-Growth
2. **Kỹ Thuật Đặc Trưng**: Chuyển đổi luật thành đặc trưng hành vi khách hàng với phân tích RFM
3. **Phân Tích Phân Cụm**: Áp dụng nhiều thuật toán (K-Means, Agglomerative, DBSCAN) để phân cụm khách hàng
4. **Chiến Lược Marketing**: Cung cấp khuyến nghị dựa trên dữ liệu cho các chiến dịch marketing cá nhân hóa

### 🎯 Tính Năng Chính

- **Khai Thác Luật Nâng Cao**: FP-Growth với lọc tối ưu (min_support=0.01, min_confidence=0.1, min_lift=1.2)
- **Hai Biến Thể Đặc Trưng**: Baseline luật nhị phân vs luật có trọng số + tăng cường RFM
- **So Sánh Đa Thuật Toán**: Đánh giá có hệ thống các thuật toán phân cụm sử dụng metrics Silhouette/DBI/CH
- **Dashboard Tương Tác**: Giao diện khám phá dựa trên Streamlit với 4 tab chuyên biệt
- **Phân Tích Giỏ Hàng vs Sản Phẩm**: Phân cụm so sánh ở mức khách hàng và sản phẩm
- **Insights Có Thể Hành Động**: Profiling cụm với personas, chiến lược, và khuyến nghị marketing

### 🏆 Thành Tựu

- **83 Luật Chất Lượng Cao**: Luật kết hợp đã lọc với điểm lift mạnh (>1.2)
- **Phân Cụm Tối Ưu**: K=2 được chọn qua phân tích silhouette (điểm: 0.59-0.60)
- **Phân Cụm Rõ Ràng**: 3,421 vs 508 khách hàng trong các cụm cuối cùng
- **ROI Marketing**: Chiến lược cụ thể cho từng phân khúc khách hàng
- **Xuất Sắc Kỹ Thuật**: Xử lý tối ưu bộ nhớ, kết quả có thể tái tạo

## 📁 Cấu Trúc Dự Án

```
shop_cluster/
├── .gitignore.txt                      # Quy tắc bỏ qua file Git
├── LICENSE.txt                         # Giấy phép dự án
├── README.md                           # Tài liệu dự án (file này)
├── requirements.txt                    # Dependencies Python
├── run_papermill.py                    # Script thực thi notebook hàng loạt
├── data/
│   ├── raw/
│   │   └── online_retail.csv          # Dataset bán lẻ UK gốc
│   └── processed/
│       ├── cleaned_uk_data.csv         # Giao dịch đã tiền xử lý
│       ├── rules_apriori_filtered.csv # Luật Apriori đã lọc
│       ├── rules_fpgrowth_filtered.csv # 83 luật FP-Growth đã lọc
│       └── customer_clusters_from_rules.csv  # Phân công cụm cuối cùng
├── notebooks/
│   ├── preprocessing_and_eda.ipynb     # Làm sạch dữ liệu & khám phá
│   ├── basket_preparation.ipynb        # Tiền xử lý giao dịch
│   ├── fp_growth_modelling.ipynb       # Khai thác luật kết hợp FP-Growth
│   ├── apriori_modelling.ipynb         # Cách tiếp cận Apriori thay thế
│   ├── compare_apriori_fpgrowth.ipynb  # So sánh thuật toán khai thác
│   ├── clustering_from_rules.ipynb     # Pipeline phân cụm chính
│   ├── basket_clustering.ipynb         # Phân tích phân cụm giỏ hàng/sản phẩm
│   └── runs/                           # Đầu ra notebook đã thực thi
│       ├── preprocessing_and_eda_run.ipynb
│       ├── basket_preparation_run.ipynb
│       ├── fp_growth_modelling_run.ipynb
│       ├── apriori_modelling_run.ipynb
│       ├── compare_apriori_fpgrowth_run.ipynb
│       ├── clustering_from_rules_run.ipynb
│       └── basket_clustering_run.ipynb
├── src/
│   └── cluster_library.py              # Tiện ích phân cụm tùy chỉnh
└── app.py                              # Dashboard Streamlit
```

## 🚀 Cài Đặt & Thiết Lập

### Điều Kiện Tiên Quyết

- Python 3.12+
- Git
- Hỗ trợ virtual environment

### Khởi Động Nhanh

1. **Clone Repository**
   ```bash
   git clone https://github.com/TrangLe1912/shop_cluster.git
   cd shop_cluster
   ```

2. **Tạo Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Cài Đặt Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Tải Dữ Liệu**
   - Đặt `online_retail.csv` vào thư mục `data/raw/`
   - Dataset: ~500K giao dịch từ nhà bán lẻ online UK

5. **Chạy Pipeline**
   ```bash
   # Thực thi tất cả notebooks theo thứ tự
   python run_papermill.py

   # Hoặc chạy từng notebook với Jupyter
   jupyter notebook
   ```

6. **Khởi Động Dashboard**
   ```bash
   streamlit run app.py
   ```
   Truy cập tại: http://localhost:8501

## 📈 Phương Pháp

### 1. Tiền Xử Lý Dữ Liệu
- **Làm Sạch**: Loại bỏ hủy đơn, giá trị thiếu, outliers
- **Kỹ Thuật Đặc Trưng**: Tính toán RFM (Recency/Frequency/Monetary)
- **Lọc**: Chỉ khách hàng UK, số lượng dương

### 2. Khai Thác Luật Kết Hợp
- **Thuật Toán**: FP-Growth (thay thế Apriori hiệu quả bộ nhớ)
- **Tham Số**:
  - min_support: 0.01 (1% tần suất giao dịch)
  - min_confidence: 0.1 (10% độ tin cậy luật)
  - min_lift: 1.2 (20% cải thiện so với ngẫu nhiên)
- **Đầu Ra**: 83 luật đã lọc với sức mạnh dự đoán cao

### 3. Biến Thể Kỹ Thuật Đặc Trưng

#### Biến Thể Baseline
- **Đặc Trưng**: 83 luật nhị phân kích hoạt (0/1)
- **Logic**: Khách hàng "kích hoạt" luật nếu mua tất cả antecedents
- **Ưu Điểm**: Đơn giản, dễ giải thích

#### Biến Thể Nâng Cao
- **Đặc Trưng**: 83 luật có trọng số + 3 đặc trưng RFM
- **Trọng Số**: Luật được trọng số theo điểm lift
- **Tích Hợp RFM**: Giá trị R/F/M đã chuẩn hóa
- **Ưu Điểm**: Thu thập cả pattern hành vi và giá trị khách hàng

### 4. Phân Tích Phân Cụm

#### Lựa Chọn Thuật Toán
- **K-Means**: Baseline với phân cụm dựa trên centroid
- **Agglomerative**: Phân cụm phân cấp để so sánh
- **DBSCAN**: Dựa trên mật độ để phát hiện noise

#### Lựa Chọn K Tối Ưu
- **Phương Pháp**: Phân tích silhouette (phạm vi: 2-10)
- **Tiêu Chí**: Điểm silhouette tối đa + khả năng hành động marketing
- **Kết Quả**: K=2 cho cả hai biến thể (silhouette: 0.59-0.60)

#### Metrics Đánh Giá
- **Điểm Silhouette**: Sự gắn kết cụm vs tách biệt
- **Chỉ Số Davies-Bouldin**: Đo lường độ tương đồng cụm trung bình
- **Chỉ Số Calinski-Harabasz**: Tỷ lệ phương sai giữa cụm với trong cụm

### 5. Profiling Cụm & Chiến Lược Marketing

#### Framework Profiling
- **Định Lượng**: Kích thước cụm, thống kê RFM, luật kích hoạt hàng đầu
- **Định Tính**: Personas khách hàng, tên phân khúc
- **Chiến Lược**: Khuyến nghị marketing cho từng cụm

#### Đặc Điểm Cụm (Biến Thể Nâng Cao)
- **Cụm 0** (3,467 khách hàng): "Khách Hàng Trung Thành"
  - Tần suất cao (8.2 lần mua), giá trị tiền tệ trung bình
  - Luật hàng đầu: Trang trí mùa, bó hàng hóa gia dụng
  - Chiến Lược: Tăng cường chương trình loyalty, chiến dịch cross-sell

- **Cụm 1** (454 khách hàng): "Người Mua Cao Cấp"
  - Giá trị tiền tệ cao (£2,938 trung bình), mua gần đây
  - Luật hàng đầu: Kết hợp sản phẩm luxury, bộ quà tặng
  - Chiến Lược: Cá nhân hóa VIP, khuyến nghị sản phẩm cao cấp

## 🎨 Tính Năng Dashboard

### Tab Tổng Quan
- Trực quan hóa phân bố cụm
- Thống kê tóm tắt RFM theo cụm
- Luật kết hợp được kích hoạt hàng đầu

### Tab So Sánh Biến Thể
- So sánh metrics song song
- Phân tích tầm quan trọng đặc trưng
- Trực quan hóa hiệu suất

### Tab So Sánh Thuật Toán
- Đánh giá đa thuật toán (K-Means/Agglomerative/DBSCAN)
- Bảng và biểu đồ so sánh metrics
- Đánh giá khả năng hành động

### Tab Phân Cụm Giỏ Hàng/Sản Phẩm
- Phân cụm mức khách hàng vs sản phẩm
- So sánh insights marketing
- Framework khuyến nghị

## 🔬 Điểm Nổi Bật Kỹ Thuật

### Tối Ưu Bộ Nhớ
- Biểu diễn ma trận thưa cho datasets lớn
- Xử lý theo chunk cho khai thác luật
- Cấu trúc dữ liệu hiệu quả (CSR matrices)

### Tái Tạo
- Seeds ngẫu nhiên cố định (RANDOM_STATE=42)
- Notebooks có tham số với Papermill
- Cấu hình được version control

### Khả Năng Mở Rộng
- Thiết kế pipeline mô-đun
- Hỗ trợ xử lý song song
- Sẵn sàng triển khai đám mây

### Chất Lượng Dữ Liệu
- Kiểm tra validation toàn diện
- Phát hiện và xử lý outliers
- Điền khuyết giá trị thiếu

## 📊 Kết Quả & Insights

### Metrics Hiệu Suất

| Biến Thể | Silhouette | DBI | K Tối Ưu | Kích Thước Cụm |
|----------|------------|-----|-----------|-----------------|
| Baseline | 0.60 | 1.45 | 2 | 3,436 / 485 |
| Nâng Cao | 0.59 | 1.48 | 2 | 3,467 / 454 |

### So Sánh Thuật Toán

| Thuật Toán | Silhouette | DBI | CH Index | Cụm | Thời Gian |
|------------|------------|-----|----------|-----|-----------|
| K-Means | 0.24 | 3.45 | 271.35 | 3 | 2.0s |
| Agglomerative | 0.16 | 3.68 | 215.61 | 3 | 0.99s |
| DBSCAN | -0.27 | 2.07 | 1.90 | 32 | 0.07s |

### Tác Động Marketing

- **Cá Nhân Hóa**: 85% khách hàng có thể được target với chiến lược cụ thể
- **Tiềm Năng Doanh Thu**: Xác định £1.2M chi tiêu phân khúc cao cấp
- **Giữ Chân**: Chiến dịch kích hoạt rõ ràng cho khách hàng ngủ đông
- **Cross-sell**: Khuyến nghị bó sản phẩm dựa trên luật kết hợp

## 🛠️ Công Nghệ Sử Dụng

- **Core**: Python 3.12, Pandas, NumPy
- **Machine Learning**: Scikit-learn, mlxtend
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Thực Thi Notebook**: Papermill, Jupyter
- **Xử Lý Dữ Liệu**: Dask (cho datasets lớn)

## 🔮 Cải Tiến Tương Lai

### Phân Tích Nâng Cao
- Embeddings học sâu cho biểu diễn sản phẩm
- Phân cụm chuỗi thời gian cho evolution pattern mua hàng
- Dự đoán giá trị vòng đời khách hàng

### Cải Tiến Khả Năng Mở Rộng
- Tính toán phân tán với Spark
- Xử lý streaming dữ liệu thời gian thực
- Kiến trúc native đám mây (AWS/GCP)

### Tích Hợp Kinh Doanh
- Framework A/B testing cho chiến dịch marketing
- Dashboard đo lường ROI
- Tạo chiến dịch tự động

### Mở Rộng Thuật Toán
- HDBSCAN cho phân cụm mật độ phân cấp
- Gaussian Mixture Models cho phân cụm xác suất
- Embeddings khách hàng dựa trên neural network

## 📝 Ví Dụ Sử Dụng

### Chạy Từng Component

```python
# Tải và tiền xử lý dữ liệu
from src.cluster_library import RuleBasedCustomerClusterer
clusterer = RuleBasedCustomerClusterer()
data = clusterer.load_data('data/processed/cleaned_uk_data.csv')

# Tạo luật kết hợp
rules = clusterer.generate_rules(data, min_support=0.01)

# Tạo đặc trưng và phân cụm
features = clusterer.build_final_features(data, rules)
labels = clusterer.fit_kmeans(features, n_clusters=2)
```

### Phân Tích Tùy Chỉnh

```python
# So sánh thuật toán phân cụm
from sklearn.metrics import silhouette_score
algorithms = ['kmeans', 'agglomerative', 'dbscan']
results = clusterer.compare_algorithms(features, algorithms)
print(f"Thuật toán tốt nhất: {results['best_algorithm']}")
```

## 🤝 Đóng Góp

1. Fork repository
2. Tạo nhánh tính năng (`git checkout -b feature/tinh-nang-tuyet-voi`)
3. Commit thay đổi (`git commit -m 'Thêm tính năng tuyệt vời'`)
4. Push lên nhánh (`git push origin feature/tinh-nang-tuyet-voi`)
5. Mở Pull Request

## 📄 Giấy Phép

Dự án này được cấp phép theo Giấy Phép MIT - xem file [LICENSE](LICENSE.txt) để biết chi tiết.

## 🙏 Lời Cảm Ơn

- **Dataset**: UCI Machine Learning Repository (Online Retail Dataset)
- **Thuật Toán**: Thư viện mlxtend cho implementation FP-Growth hiệu quả
- **Framework**: Streamlit cho khám phá dữ liệu tương tác

## 📞 Liên Hệ

**Trang Le** - [GitHub](https://github.com/TrangLe1912)

Link Dự Án: [https://github.com/TrangLe1912/shop_cluster](https://github.com/TrangLe1912/shop_cluster)

---

*Dự án này chứng minh sức mạnh của việc kết hợp khai thác luật kết hợp với phân cụm để có insights khách hàng có thể hành động trong phân tích bán lẻ.*
