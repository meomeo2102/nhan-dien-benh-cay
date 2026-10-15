import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. TẢI MÔ HÌNH AI (Đã huấn luyện 39 classes)
# ==========================================
model = tf.keras.models.load_model('best_plant_model.keras')

# ==========================================
# 2. KHAI BÁO DANH SÁCH 39 LỚP (Có lớp Ảnh Rác)
# ==========================================
class_names = [
    'Táo - Bệnh vảy đen', 'Táo - Bệnh thối đen', 'Táo - Bệnh gỉ sắt', 'Táo - Khỏe mạnh',
    'Việt quất - Khỏe mạnh', 'Anh đào - Bệnh phấn trắng', 'Anh đào - Khỏe mạnh',
    'Ngô - Bệnh đốm xám', 'Ngô - Bệnh gỉ sắt', 'Ngô - Bệnh cháy lá', 'Ngô - Khỏe mạnh',
    'Nho - Bệnh thối đen', 'Nho - Bệnh sởi đen', 'Nho - Bệnh đốm lá', 'Nho - Khỏe mạnh',
    'Cam/Chanh - Bệnh vàng lá gân xanh', 'Đào - Bệnh đốm vi khuẩn', 'Đào - Khỏe mạnh',
    'Ớt chuông - Bệnh đốm vi khuẩn', 'Ớt chuông - Khỏe mạnh', 'Khoai tây - Bệnh sương mai sớm',
    'Khoai tây - Bệnh sương mai muộn', 'Khoai tây - Khỏe mạnh', 'Mâm xôi - Khỏe mạnh',
    'Đậu nành - Khỏe mạnh', 'Bí đỏ - Bệnh phấn trắng', 'Dâu tây - Bệnh cháy lá', 'Dâu tây - Khỏe mạnh',
    'Cà chua - Bệnh đốm vi khuẩn', 'Cà chua - Bệnh đốm vòng', 'Cà chua - Bệnh sương mai muộn',
    'Cà chua - Bệnh nấm lá', 'Cà chua - Bệnh đốm lá Septoria', 'Cà chua - Nhện đỏ', 'Cà chua - Bệnh đốm đích',
    'Cà chua - Bệnh xoăn vàng lá', 'Cà chua - Bệnh khảm', 'Cà chua - Khỏe mạnh',
    'ZZ_Khong_Phai_La_Cay' # LỚP THỨ 39: BỘ LỌC CHỐNG NHẬN DIỆN NHẦM
]

# ==========================================
# 3. LUỒNG XỬ LÝ VÀ CHẨN ĐOÁN
# ==========================================
def predict_disease(img):
    if img is None:
        return {"Vui lòng chụp hoặc tải lên một hình ảnh.": 1.0}

    try:
        # Tiền xử lý: Ép về hệ màu RGB 3 kênh và resize chuẩn 224x224
        img_resized = img.convert("RGB").resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        # Suy luận (Inference) ra 39 xác suất
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)

        # ---------------------------------------------------------
        # BỘ LỌC OOD (Out-of-Distribution)
        # ---------------------------------------------------------
        # 1. Bắt trúng ảnh rác đồ vật/môi trường
        if class_names[predicted_idx] == 'ZZ_Khong_Phai_La_Cay':
            return {"⚠️ Lỗi: Hình ảnh không hợp lệ (Không phải lá cây nông nghiệp). Vui lòng thử lại!": 1.0}

        # 2. Tín hiệu quá nhiễu, AI không chắc chắn
        if predictions[predicted_idx] < 0.45:
            return {"⚠️ Cảnh báo: Hình ảnh quá mờ, nhiễu hoặc hệ thống không chắc chắn.": 1.0}

        # Đóng gói kết quả: Chỉ lấy 38 lớp bệnh (bỏ qua lớp 39) để đẩy lên giao diện
        confidences = {class_names[i]: float(predictions[i]) for i in range(38)}
        return confidences

    except Exception as e:
        return {f"⚠️ Đã xảy ra lỗi hệ thống khi xử lý ảnh: {str(e)}": 1.0}

# ==========================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI GRADIO)
# ==========================================
custom_css = """
    .gradio-container { font-family: 'Arial', sans-serif; }
    h1 { color: #2e7d32; text-align: center; }
    p { text-align: center; font-size: 16px; }
    footer { display: none !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate")) as demo:
    gr.Markdown("# 🌱 HỆ THỐNG TRÍ TUỆ NHÂN TẠO CHẨN ĐOÁN BỆNH CÂY TRỒNG")
    gr.Markdown("**Hướng dẫn:** Chụp trực tiếp bằng Camera hoặc tải ảnh lá cây lên. "
                "Hệ thống AI (MobileNetV2) sẽ phân tích và trả về 3 kết quả có khả năng cao nhất.")

    with gr.Tabs():
        with gr.Tab("📷 Chụp bằng Camera"):
            cam_input = gr.Image(type="pil", sources=["webcam"], label="Khung Camera")
            cam_btn = gr.Button("🔍 Tiến hành Chẩn Đoán", variant="primary")

        with gr.Tab("🖼️ Tải ảnh / Chụp bằng điện thoại"):
            up_input = gr.Image(type="pil", sources=["upload"], label="Khung Tải Ảnh")
            up_btn = gr.Button("🔍 Tiến hành Chẩn Đoán", variant="primary")

    label_output = gr.Label(num_top_classes=3, label="Kết quả phân tích")

    cam_btn.click(fn=predict_disease, inputs=cam_input, outputs=label_output)
    up_btn.click(fn=predict_disease, inputs=up_input, outputs=label_output)

# ==========================================
# 5. KHỞI CHẠY MÁY CHỦ
# ==========================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
