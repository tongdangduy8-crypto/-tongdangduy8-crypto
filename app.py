import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Hệ thống Phân loại Email Spam",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo tiêu đề ứng dụng
st.title("📧 Hệ thống Phân loại Email Spam sử dụng Cây Quyết Định")
st.markdown("---")

# Sidebar điều hướng
st.sidebar.title("Điều Hướng Hệ Thống")
menu = st.sidebar.radio(
    "Chọn chức năng:",
    ["1. Hướng dẫn & Tải dữ liệu", "2. Huấn luyện & Đánh giá", "3. Dự đoán Email mới"]
)

# Hàm cache để đọc dữ liệu
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Lỗi định dạng file: {e}")
        return None

# Hàm cache huấn luyện mô hình
@st.cache_resource
def train_model(X_train, y_train, max_depth=3):
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Khởi tạo session state để lưu trữ dữ liệu và mô hình xuyên suốt các trang
if 'df' not in st.session_state:
    st.session_state.df = None
if 'clf' not in st.session_state:
    st.session_state.clf = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'imputer' not in st.session_state:
    st.session_state.imputer = None

# --- TRANG 1: HƯỚNG DẪN & TẢI DỮ LIỆU ---
if menu == "1. Hướng dẫn & Tải dữ liệu":
    st.header("📁 Tải dữ liệu huấn luyện")
    st.markdown("""
    Chào mừng bạn đến với ứng dụng **Phân loại Email Spam trực tuyến**. Để hệ thống hoạt động, vui lòng chuẩn bị và tải lên file dữ liệu Excel của bạn.
    
    **Yêu cầu định dạng file Excel (.xlsx):**
    - Phải có cột **`So_tu_khoa_quang_cao`** (Số nguyên)
    - Phải có cột **`So_lien_ket`** (Số nguyên)
    - Phải có cột mục tiêu **`Spam`** (Nhận giá trị `0` - Không Spam, hoặc `1` - Spam)
    """)
    
    uploaded_file = st.file_uploader("Chọn tệp Excel chứa dữ liệu của bạn", type=["xlsx"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            # Kiểm tra xem các cột cần thiết có tồn tại không
            required_cols = ['So_tu_khoa_quang_cao', 'So_lien_ket', 'Spam']
            if all(col in df.columns for col in required_cols):
                st.session_state.df = df
                st.success("🎉 Tải dữ liệu và đọc thành công!")
                
                # Hiển thị thông tin dữ liệu
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Xem nhanh 5 hàng đầu tiên")
                    st.dataframe(df.head())
                with col2:
                    st.subheader("Thống kê số lượng phân lớp")
                    st.write(f"**Tổng số mẫu:** {len(df)}")
                    st.dataframe(df['Spam'].value_counts().rename(index={0: 'Không Spam (0)', 1: 'Spam (1)'}))
            else:
                st.error(f"Tệp Excel thiếu một trong các cột bắt buộc sau: {required_cols}")

# --- TRANG 2: HUẤN LUYỆN & ĐÁNH GIÁ ---
elif menu == "2. Huấn luyện & Đánh giá":
    st.header("⚙️ Huấn luyện và Đánh giá Mô hình")
    
    if st.session_state.df is None:
        st.warning("⚠️ Vui lòng tải dữ liệu Excel lên tại Mục **'1. Hướng dẫn & Tải dữ liệu'** trước.")
    else:
        df = st.session_state.df
        X = df[['So_tu_khoa_quang_cao', 'So_lien_ket']]
        y = df['Spam']
        
        st.markdown("### Cấu hình tham số Cây Quyết Định")
        max_depth = st.slider("Độ sâu tối đa của cây (max_depth):", min_value=1, max_value=10, value=3)
        
        # Tiến hành phân chia và tiền xử lý dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        imputer = SimpleImputer(strategy='median')
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        
        # Huấn luyện mô hình
        clf = train_model(X_train_imp, y_train, max_depth)
        
        # Lưu lại vào session_state
        st.session_state.clf = clf
        st.session_state.X_train = X_train_imp
        st.session_state.X_test = X_test_imp
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.imputer = imputer
        
        st.success("🎉 Đã hoàn tất huấn luyện mô hình cây quyết định!")
        
        # Hiển thị biểu đồ cây quyết định
        st.subheader("🌳 Sơ đồ Cây Quyết Định (Decision Tree Plot)")
        fig_tree, ax_tree = plt.subplots(figsize=(10, 6))
        plot_tree(clf, 
                  feature_names=list(X_train_imp.columns), 
                  class_names=['Không Spam', 'Spam'], 
                  filled=True, 
                  rounded=True, 
                  fontsize=10, 
                  ax=ax_tree)
        st.pyplot(fig_tree)
        
        # Các thông số đánh giá trên tập Test
        y_pred = clf.predict(X_test_imp)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Hiển thị các chỉ số đo lường hiệu năng
        st.subheader("📊 Chỉ số hiệu năng (Trên tập dữ liệu kiểm thử - Test)")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{accuracy:.4f}")
        m_col2.metric("Precision (Spam)", f"{precision:.4f}")
        m_col3.metric("Recall (Spam)", f"{recall:.4f}")
        m_col4.metric("F1-score", f"{f1:.4f}")
        
        # Hiển thị Ma trận nhầm lẫn và Biểu đồ phân vùng
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🌀 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Không Spam', 'Spam'])
            disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d')
            st.pyplot(fig_cm)
            
        with col_right:
            st.subheader("🌈 Biểu đồ phân vùng ranh giới quyết định")
            try:
                x_min, x_max = X_train_imp.iloc[:, 0].min() - 1, X_train_imp.iloc[:, 0].max() + 1
                y_min, y_max = X_train_imp.iloc[:, 1].min() - 1, X_train_imp.iloc[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                fig_bound, ax_bound = plt.subplots(figsize=(6, 5))
                ax_bound.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
                scatter = ax_bound.scatter(X_train_imp.iloc[:, 0], X_train_imp.iloc[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
                ax_bound.set_xlabel('Số từ khóa quảng cáo')
                ax_bound.set_ylabel('Số liên kết')
                legend = ax_bound.legend(*scatter.legend_elements(), title="Nhãn")
                legend.get_texts()[0].set_text('Không Spam')
                legend.get_texts()[1].set_text('Spam')
                st.pyplot(fig_bound)
            except Exception as e:
                st.info("Không thể vẽ biểu đồ phân vùng (Dữ liệu không đủ đa dạng hoặc lỗi vẽ lưới)")

# --- TRANG 3: DỰ ĐOÁN EMAIL MỚI ---
elif menu == "3. Dự đoán Email mới":
    st.header("🔮 Kiểm tra phân loại Email mới")
    
    if st.session_state.clf is None:
        st.warning("⚠️ Bạn cần thực hiện huấn luyện mô hình thành công tại Mục **'2. Huấn luyện & Đánh giá'** trước.")
    else:
        clf = st.session_state.clf
        imputer = st.session_state.imputer
        
        st.markdown("Nhập các thông số của email nhận được để phân loại tự động:")
        
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            x1_val = st.number_input("Số từ khóa quảng cáo xuất hiện:", min_value=0, value=0, step=1)
        with col_in2:
            x2_val = st.number_input("Số liên kết (links) đính kèm trong email:", min_value=0, value=0, step=1)
            
        if st.button("💡 Tiến hành phân tích", type="primary"):
            # Đóng gói dữ liệu đầu vào
            sample = np.array([[x1_val, x2_val]])
            # Áp dụng imputer
            sample_imputed = imputer.transform(sample)
            
            pred_label = clf.predict(sample_imputed)[0]
            pred_prob = clf.predict_proba(sample_imputed)[0, 1]
            
            label_text = "🚨 EMAIL SPAM (RÁC)" if pred_label == 1 else "✅ EMAIL AN TOÀN (KHÔNG SPAM)"
            color = "red" if pred_label == 1 else "green"
            
            st.markdown(f"### Kết quả phân loại: <span style='color:{color}'>{label_text}</span>", unsafe_allow_html=True)
            st.markdown(f"- **Xác suất là Email Spam:** `{pred_prob * 100:.2f}%`")
            
            # Minh họa đường đi quyết định
            st.markdown("#### 🔍 Giải thích quyết định từ Cây Quyết Định:")
            node_indicator = clf.decision_path(sample_imputed)
            feature = clf.tree_.feature
            threshold = clf.tree_.threshold
            feature_names = ['Số từ khóa quảng cáo', 'Số liên kết']
            
            node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
            for idx, node_id in enumerate(node_index):
                if feature[node_id] != -2:
                    feat_name = feature_names[feature[node_id]]
                    val = sample_imputed[0, feature[node_id]]
                    direction = "<=" if val <= threshold[node_id] else ">"
                    st.write(f"- **Bước {idx + 1}**: Xét {feat_name} (`{val}`) {direction} `{threshold[node_id]:.2f}`")
                else:
                    st.markdown(f"- 🎯 **Quyết định tại nút lá số {node_id}**")
