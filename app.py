
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Fruit Detection System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - DARK PREMIUM DESIGN
# =====================================================

st.markdown("""
<style>

/* MAIN APPLICATION */
.stApp {
    background-color: #0b1120;
    color: #f8fafc;
    font-family: 'Segoe UI', sans-serif;
}

/* MAIN CONTENT */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263244;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* SIDEBAR HEADER */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #34d399;
}

/* MAIN HEADINGS */
h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700;
}

/* DASHBOARD HEADER */
.dashboard-header {
    background: linear-gradient(135deg, #172554, #064e3b);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #166534;
    margin-bottom: 25px;
}

.dashboard-header h1 {
    color: #ffffff !important;
    font-size: 34px;
    margin-bottom: 8px;
}

.dashboard-header p {
    color: #a7f3d0;
    font-size: 15px;
}

/* SECTION TITLES */
.section-heading {
    font-size: 21px;
    font-weight: 700;
    color: #34d399;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* GROUP MEMBERS CARD */
.members-card {
    background: #151f30;
    border: 1px solid #26364b;
    border-left: 4px solid #10b981;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
}

.members-card p {
    color: #d1d5db;
    margin: 10px 0;
    font-size: 14px;
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: #151f30;
    border: 1px solid #26364b;
    padding: 22px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
}

[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
}

[data-testid="stMetricValue"] {
    color: #34d399 !important;
    font-weight: 700;
}

/* UPLOAD AREA */
[data-testid="stFileUploader"] {
    background: #151f30;
    border: 1px dashed #10b981;
    border-radius: 15px;
    padding: 20px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

/* BUTTONS */
.stButton > button {
    background: #059669;
    color: white;
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    background: #047857;
    border-color: #34d399;
    color: white;
}

/* DOWNLOAD BUTTONS */
.stDownloadButton > button {
    background: #1f2937;
    color: #34d399;
    border: 1px solid #10b981;
    border-radius: 10px;
    font-weight: 600;
    padding: 12px;
}

.stDownloadButton > button:hover {
    background: #064e3b;
    color: white;
    border-color: #34d399;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 1px solid #26364b;
    border-radius: 12px;
}

/* IMAGE */
[data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
}

/* SLIDER */
[data-testid="stSlider"] {
    padding-top: 10px;
    padding-bottom: 10px;
}

/* DIVIDER */
hr {
    border-color: #263244;
}

/* INFO AND WARNING */
.stAlert {
    border-radius: 12px;
}

/* FOOTER */
.dashboard-footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 25px;
    border-top: 1px solid #263244;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# DASHBOARD HEADER
# =====================================================

st.markdown("""
<div class="dashboard-header">
    <h1>🍎 Fruit Detection System</h1>
    <p>AI-powered fruit detection using YOLOv8n.</p>
    <p>Computer Vision | Artificial Intelligence | Object Detection</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# GROUP MEMBERS
# =====================================================

st.markdown(
    '<div class="section-heading">👥 GROUP MEMBERS</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="members-card">
    <p><b>01.</b> MFITUMUKIZA Deogratias 26RP02925</p>
    <p><b>02.</b> UZAYISENGA Marie Bella 26RP02449</p>
    <p><b>03.</b> NDONGOZI Chris Arsene 26RP00610</p>
    <p><b>04.</b> UMWIZA Alima 26RP01729</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# CLASS MAPPING
# =====================================================

CLASS_NAMES = {
    1: "apple",
    2: "banana",
    3: "orange",
}

COLORS = {
    "apple": (255, 0, 0),
    "banana": (255, 200, 0),
    "orange": (0, 180, 0),
}


# =====================================================
# LOAD YOLO MODEL
# =====================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


model = load_model()


# =====================================================
# SIDEBAR SETTINGS
# =====================================================

st.sidebar.markdown("""
<h2>🍎 FRUIT AI</h2>
<p style="color:#9ca3af;">Detection Dashboard</p>
<hr>
""", unsafe_allow_html=True)

st.sidebar.subheader("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a fruit image and use the AI model to detect apples, bananas and oranges."
)

st.sidebar.markdown("---")

st.sidebar.caption("YOLO Object Detection System")


# =====================================================
# IMAGE UPLOAD
# =====================================================

st.markdown(
    '<div class="section-heading">📤 Upload Fruit Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)


# =====================================================
# IMAGE PROCESSING
# =====================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")

    col1, col2 = st.columns([1.5, 1])

    with col1:

        st.markdown("### 🖼️ Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### 🔍 Detection")

        st.write("Your image is ready for AI analysis.")

        st.write(
            f"Confidence threshold: {confidence:.2f}"
        )

        detect_button = st.button(
            "🔍 Detect Fruits",
            type="primary",
            use_container_width=True
        )


    # =================================================
    # YOLO DETECTION
    # =================================================

    if detect_button:

        with st.spinner("Analyzing image..."):

            results = model.predict(
                image,
                conf=confidence,
                iou=0.45,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            annotated_image = result.plot()

            annotated_image = Image.fromarray(
                annotated_image[:, :, ::-1]
            )

            fruit_counts = {
                "apple": 0,
                "banana": 0,
                "orange": 0
            }

            detection_data = []

            if boxes is not None and len(boxes) > 0:

                for box in boxes:

                    class_id = int(box.cls[0])

                    fruit = model.names[class_id].lower()

                    score = float(box.conf[0])

                    if fruit not in fruit_counts:
                        continue

                    fruit_counts[fruit] += 1

                    detection_data.append({
                        "Fruit": fruit.capitalize(),
                        "Confidence": round(score * 100, 2)
                    })


        # =============================================
        # DETECTION RESULTS
        # =============================================

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">📸 Detection Results</div>',
            unsafe_allow_html=True
        )

        st.image(
            annotated_image,
            caption="YOLO Fruit Detection Output",
            use_container_width=True
        )


        # =============================================
        # FRUIT SUMMARY
        # =============================================

        total = sum(fruit_counts.values())

        st.markdown(
            '<div class="section-heading">📊 Fruit Detection Summary</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Fruits",
            total
        )

        col2.metric(
            "🍎 Apples",
            fruit_counts["apple"]
        )

        col3.metric(
            "🍌 Bananas",
            fruit_counts["banana"]
        )

        col4.metric(
            "🍊 Oranges",
            fruit_counts["orange"]
        )


        # =============================================
        # DETECTION DETAILS
        # =============================================

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">📋 Detection Details</div>',
            unsafe_allow_html=True
        )

        if detection_data:

            df = pd.DataFrame(detection_data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # CSV DOWNLOAD

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            image_buffer = io.BytesIO()

            annotated_image.save(
                image_buffer,
                format="PNG"
            )

            st.markdown("### 📥 Download Results")

            download_col1, download_col2 = st.columns(2)

            with download_col1:

                st.download_button(
                    "📄 Download Detection Results (CSV)",
                    data=csv,
                    file_name="fruit_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with download_col2:

                st.download_button(
                    "🖼️ Download Annotated Image",
                    data=image_buffer.getvalue(),
                    file_name="fruit_detection_result.png",
                    mime="image/png",
                    use_container_width=True
                )

        else:

            st.warning(
                "No fruits detected. Try lowering the confidence threshold."
            )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="dashboard-footer">
    🍎 FRUIT DETECTION SYSTEM |
    AI-Powered Object Detection
</div>
""", unsafe_allow_html=True)
