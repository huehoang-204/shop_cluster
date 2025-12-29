import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Cấu hình trang
st.set_page_config(
    page_title="Dashboard Phân Cụm Khách Hàng",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .explanation-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .highlight-box {
        background: #fff3e0;
        border: 2px solid #ff9800;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    clusters = pd.read_csv("data/processed/customer_clusters_from_rules.csv")
    rules = pd.read_csv("data/processed/rules_fpgrowth_filtered.csv")
    return clusters, rules

clusters, rules = load_data()

# Header chính
st.markdown('<h1 class="main-header">🛍️ Dashboard Phân Cụm Khách Hàng Bán Lẻ</h1>', unsafe_allow_html=True)

# Sidebar điều hướng
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("🚀 Điều Hướng")

    # Thông tin tổng quan
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👥 Khách hàng", f"{len(clusters):,}")
    with col2:
        st.metric("📊 Cụm", len(clusters['cluster'].unique()))

    st.markdown("---")

    # Menu chính
    page = st.radio(
        "Chọn trang:",
        ["📈 Tổng Quan", "🔄 So Sánh Biến Thể", "🤖 So Sánh Thuật Toán", "🛒 Phân Cụm Giỏ/Sản Phẩm"],
        index=0
    )

    st.markdown("---")

    # Thông tin project
    with st.expander("ℹ️ Về Dự Án"):
        st.markdown("""
        **Dự án phân cụm khách hàng** sử dụng:
        - Luật kết hợp (FP-Growth)
        - Phân cụm K-Means
        - Phân tích RFM
        - Dashboard tương tác
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# Trang Tổng Quan
if page == "📈 Tổng Quan":
    st.header("📊 Tổng Quan Phân Cụm")

    # Metrics chính
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">3,921</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Tổng Khách Hàng</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">2</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Số Cụm</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">83</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Luật Kết Hợp</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">0.59</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Silhouette Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bộ lọc và chi tiết cụm
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Chọn Cụm")
        selected_cluster = st.selectbox(
            "Chọn cụm để xem chi tiết:",
            options=sorted(clusters['cluster'].unique()),
            format_func=lambda x: f"Cụm {x}"
        )

        # Thông tin cụm
        filtered_clusters = clusters[clusters['cluster'] == selected_cluster]
        st.metric("👥 Số khách hàng", len(filtered_clusters))

        # Phân tích RFM nếu có
        if 'Recency' in filtered_clusters.columns:
            st.subheader("📊 RFM Analysis")
            rfm_stats = filtered_clusters[['Recency', 'Frequency', 'Monetary']].describe()

            # Hiển thị metrics RFM
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.metric("Recency (ngày)", f"{rfm_stats.loc['mean', 'Recency']:.0f}")
            with r_col2:
                st.metric("Frequency", f"{rfm_stats.loc['mean', 'Frequency']:.1f}")
            with r_col3:
                st.metric("Monetary (£)", f"{rfm_stats.loc['mean', 'Monetary']:.0f}")

    with col2:
        # Phân bố cụm
        st.subheader("📈 Phân Bố Các Cụm")

        cluster_counts = clusters['cluster'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cụm', 'y': 'Số khách hàng'},
            title="Phân bố khách hàng theo cụm",
            color=cluster_counts.index,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Thông tin chi tiết cụm được chọn
        st.subheader(f"📋 Chi Tiết Cụm {selected_cluster}")
        cluster_info = {
            "Tên cụm": ["Khách hàng trung thành", "Khách hàng cao cấp"][selected_cluster],
            "Kích thước": len(filtered_clusters),
            "Tỷ lệ": f"{len(filtered_clusters)/len(clusters)*100:.1f}%",
            "Đặc điểm": [
                "Mua thường xuyên, giá trị trung bình",
                "Mua ít nhưng giá trị cao"
            ][selected_cluster]
        }

        info_df = pd.DataFrame(list(cluster_info.items()), columns=['Thuộc tính', 'Giá trị'])
        st.dataframe(info_df, use_container_width=True)

    # Luật kết hợp hàng đầu
    st.header("🔗 Top Luật Kết Hợp")
    with st.expander("ℹ️ Giải thích về Luật Kết Hợp"):
        st.markdown("""
        **Luật kết hợp** cho thấy mối quan hệ giữa các sản phẩm:
        - **Support**: Tần suất xuất hiện của luật
        - **Confidence**: Độ tin cậy khi có A thì có B
        - **Lift**: Độ mạnh của mối quan hệ (>1 là có ý nghĩa)
        """)

    # Hiển thị top rules
    top_rules = rules.head(10)[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
    top_rules.columns = ['Điều kiện (A)', 'Kết quả (B)', 'Support', 'Confidence', 'Lift']

    st.dataframe(
        top_rules.style.format({
            'Support': '{:.3f}',
            'Confidence': '{:.3f}',
            'Lift': '{:.2f}'
        }),
        use_container_width=True
    )

    # Gợi ý bundle
    st.header("🎁 Gợi Ý Bundle & Cross-sell")
    with st.expander("💡 Cách sử dụng gợi ý"):
        st.markdown("""
        Dựa trên luật kết hợp, chúng ta có thể:
        - **Bundle**: Gói sản phẩm mua kèm nhau
        - **Cross-sell**: Đề xuất sản phẩm liên quan
        - **Upsell**: Nâng cấp lên sản phẩm cao cấp hơn
        """)

    bundle_suggestions = [
        "🎄 Đèn trang trí + Vỏ gối → Bộ quà Giáng sinh",
        "☕ Tách trà + Đèn bàn → Bộ dụng cụ pha trà",
        "🧸 Đồ chơi trẻ em + Sách → Bộ quà tặng trẻ em",
        "🍽️ Đũa + Tấm trải bàn → Bộ dụng cụ ăn uống",
        "🕯️ Nến thơm + Khay đựng → Bộ trang trí nhà cửa"
    ]

    for suggestion in bundle_suggestions:
        st.markdown(f"• {suggestion}")

# Trang So Sánh Biến Thể
elif page == "🔄 So Sánh Biến Thể":
    st.header("🔄 So Sánh Các Biến Thể Đặc Trưng")

    with st.expander("📖 Giải thích về các biến thể"):
        st.markdown("""
        **Baseline (Cơ bản)**: Chỉ sử dụng luật nhị phân (0/1)
        - Ưu điểm: Đơn giản, dễ hiểu
        - Nhược điểm: Không phân biệt độ mạnh yếu của luật

        **Advanced (Nâng cao)**: Luật có trọng số + RFM
        - Luật được trọng số theo lift
        - Thêm 3 đặc trưng RFM (Recency/Frequency/Monetary)
        - Ưu điểm: Chính xác hơn, phân tích toàn diện hơn
        """)

    # Bảng so sánh
    comparison_data = {
        'Biến Thể': ['Baseline (Luật nhị phân)', 'Advanced (Luật trọng số + RFM)'],
        'Số Đặc Trưng': ['83 (luật 0/1)', '86 (83 luật + 3 RFM)'],
        'Silhouette Score': [0.60, 0.59],
        'K Tối Ưu': [2, 2],
        'Cụm 0': ['3,436 KH', '3,467 KH'],
        'Cụm 1': ['485 KH', '454 KH'],
        'Ưu Điểm': ['Đơn giản, nhanh', 'Chi tiết hơn, phân tích RFM tốt']
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Hiển thị bảng
    st.dataframe(comparison_df, use_container_width=True)

    # Biểu đồ so sánh
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 So Sánh Silhouette Score")
        fig = px.bar(
            comparison_df,
            x='Biến Thể',
            y='Silhouette Score',
            title="Điểm Silhouette theo biến thể",
            color='Biến Thể',
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Phân Bố Kích Thước Cụm")
        # Tạo dữ liệu cho stacked bar
        size_data = pd.DataFrame({
            'Biến Thể': ['Baseline', 'Baseline', 'Advanced', 'Advanced'],
            'Cụm': ['Cụm 0', 'Cụm 1', 'Cụm 0', 'Cụm 1'],
            'Số KH': [3436, 485, 3467, 454]
        })

        fig = px.bar(
            size_data,
            x='Biến Thể',
            y='Số KH',
            color='Cụm',
            title="Kích thước cụm theo biến thể",
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Kết luận
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("""
    **🎯 Kết Luận:**
    - Biến thể Advanced cho kết quả tương tự nhưng chi tiết hơn
    - Việc thêm RFM giúp phân tích giá trị khách hàng tốt hơn
    - Cả hai biến thể đều tạo ra 2 cụm có ý nghĩa marketing
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Trang So Sánh Thuật Toán
elif page == "🤖 So Sánh Thuật Toán":
    st.header("🤖 So Sánh Các Thuật Toán Phân Cụm")

    with st.expander("📚 Giải thích các thuật toán"):
        st.markdown("""
        **K-Means**: Thuật toán centroid-based, nhanh và hiệu quả
        - Ưu điểm: Tốc độ cao, dễ hiểu
        - Nhược điểm: Giả định cụm hình cầu

        **Agglomerative**: Phân cụm phân cấp, xây dựng cây cụm
        - Ưu điểm: Không cần chỉ định K trước, linh hoạt
        - Nhược điểm: Chậm với dữ liệu lớn

        **DBSCAN**: Dựa trên mật độ, phát hiện cụm bất kỳ hình dạng
        - Ưu điểm: Tự động phát hiện noise, không cần K
        - Nhược điểm: Nhạy cảm với tham số eps và min_samples
        """)

    # Chạy so sánh thuật toán
    if 'Recency' in clusters.columns:
        features = clusters[['Recency', 'Frequency', 'Monetary']].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        algorithms = {
            'K-Means': None,  # Đã có sẵn
            'Agglomerative': AgglomerativeClustering(n_clusters=2),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (name, algo) in enumerate(algorithms.items()):
            status_text.text(f"Đang chạy {name}...")
            progress_bar.progress((i + 1) / len(algorithms))

            if name == 'K-Means':
                labels = clusters['cluster'].values
            else:
                labels = algo.fit_predict(X_scaled)
                if name == 'DBSCAN':
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    st.info(f"{name}: Tìm thấy {n_clusters} cụm")

            if len(set(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
                dbi = davies_bouldin_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)
                results.append({
                    'Thuật Toán': name,
                    'Silhouette': sil,
                    'DBI': dbi,
                    'CH': ch,
                    'Số Cụm': len(set(labels))
                })
            else:
                results.append({
                    'Thuật Toán': name,
                    'Silhouette': 'N/A',
                    'DBI': 'N/A',
                    'CH': 'N/A',
                    'Số Cụm': 1
                })

        progress_bar.empty()
        status_text.empty()

        results_df = pd.DataFrame(results)

        # Hiển thị kết quả
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Bảng Kết Quả")
            st.dataframe(
                results_df.style.format({
                    'Silhouette': '{:.3f}',
                    'DBI': '{:.3f}',
                    'CH': '{:.1f}'
                }),
                use_container_width=True
            )

        with col2:
            st.subheader("📈 Biểu Đồ So Sánh")
            fig = px.bar(
                results_df,
                x='Thuật Toán',
                y='Silhouette',
                title="So sánh Silhouette Score",
                color='Thuật Toán',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, use_container_width=True)

        # Phân tích chi tiết
        st.subheader("🔍 Phân Tích Chi Tiết")

        best_algo = results_df.loc[results_df['Silhouette'].idxmax(), 'Thuật Toán']
        st.success(f"🎯 Thuật toán tốt nhất: **{best_algo}** (Silhouette cao nhất)")

        # Giải thích metrics
        with st.expander("📖 Giải thích Metrics"):
            st.markdown("""
            **Silhouette Score**: Đo lường chất lượng phân cụm
            - Gần 1: Cụm tốt, điểm cách xa centroid khác
            - Gần 0: Điểm ở biên giới cụm
            - Âm: Điểm có thể ở cụm sai

            **DBI (Davies-Bouldin Index)**: Trung bình tỷ lệ similarity
            - Thấp tốt: Cụm compact và well-separated

            **CH (Calinski-Harabasz)**: Tỷ lệ between/within variance
            - Cao tốt: Cụm distinct và compact
            """)

# Trang Phân Cụm Giỏ/Sản Phẩm
elif page == "🛒 Phân Cụm Giỏ/Sản Phẩm":
    st.header("🛒 So Sánh Phân Cụm Giỏ Hàng vs Sản Phẩm")

    with st.expander("🎯 Mục đích phân tích"):
        st.markdown("""
        **Phân cụm ở 2 góc nhìn khác nhau:**
        - **Giỏ hàng (Basket)**: Nhóm khách hàng theo pattern mua hàng
        - **Sản phẩm (Product)**: Nhóm sản phẩm theo sự tương đồng

        **Ứng dụng marketing:**
        - Basket clustering: Phân khúc khách hàng, personalized marketing
        - Product clustering: Gợi ý sản phẩm, tối ưu layout cửa hàng
        """)

    # Kết quả Basket Clustering
    st.subheader("🛒 Basket Clustering (Khách hàng)")

    basket_results = {
        'Thuật Toán': ['K-Means', 'Agglomerative', 'DBSCAN'],
        'Silhouette': [0.24, 0.16, -0.27],
        'DBI': [3.45, 3.68, 2.07],
        'CH': [271.35, 215.61, 1.90],
        'Số Cụm': [3, 3, 32],
        'Thời Gian (s)': [2.00, 0.99, 0.07]
    }

    basket_df = pd.DataFrame(basket_results)
    st.dataframe(basket_df, use_container_width=True)

    # Kết quả Product Clustering
    st.subheader("📦 Product Clustering (Sản phẩm)")

    product_results = {
        'Phương Pháp': ['Product Clustering'],
        'Silhouette': [0.35],
        'DBI': [1.10],
        'CH': [83.43],
        'Số Cụm': [5],
        'Thuật Toán': ['K-Means trên ma trận tương đồng']
    }

    product_df = pd.DataFrame(product_results)
    st.dataframe(product_df, use_container_width=True)

    # So sánh trực quan
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Basket vs Product")
        comparison_data = {
            'Tiêu Chí': ['Input', 'Output', 'Ứng Dụng Marketing', 'Ưu Điểm'],
            'Basket Clustering': [
                '3,549 khách × 100 sản phẩm',
                '3-32 cụm khách hàng',
                'Phân khúc KH, personalized',
                'Trực tiếp actionable'
            ],
            'Product Clustering': [
                '100 sản phẩm × 100 sản phẩm',
                '5 cụm sản phẩm',
                'Gợi ý sản phẩm, layout',
                'Tự động hóa recommendations'
            ]
        }

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True)

    with col2:
        st.subheader("🎯 Khuyến Nghị")
        st.markdown("""
        **Dựa trên kết quả:**

        🏆 **Basket Clustering** được khuyến nghị cho:
        - Phân tích giá trị khách hàng
        - Chiến lược marketing cá nhân hóa
        - Tăng trưởng doanh thu từ khách hàng hiện hữu

        📦 **Product Clustering** phù hợp cho:
        - Hệ thống gợi ý sản phẩm
        - Tối ưu layout cửa hàng
        - Phân tích danh mục sản phẩm
        """)

        # Biểu đồ radar comparison
        categories = ['Silhouette', 'DBI', 'CH Score', 'Actionability']
        basket_scores = [0.24, 3.45, 271.35, 8]  # Normalized
        product_scores = [0.35, 1.10, 83.43, 7]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=basket_scores,
            theta=categories,
            fill='toself',
            name='Basket Clustering'
        ))
        fig.add_trace(go.Scatterpolar(
            r=product_scores,
            theta=categories,
            fill='toself',
            name='Product Clustering'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 300])),
            showlegend=True,
            title="So sánh Basket vs Product Clustering"
        )

        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🛍️ <strong>Dashboard Phân Cụm Khách Hàng</strong> | Dự án Data Mining 2025</p>
    <p>Được xây dựng với ❤️ sử dụng Streamlit</p>
</div>
""", unsafe_allow_html=True)