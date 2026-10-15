import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import os

# ==========================================
# 1. TẢI MÔ HÌNH AI (39 Classes)
# ==========================================
model = tf.keras.models.load_model('best_plant_model.keras')

# ==========================================
# 2. KHAI BÁO DANH SÁCH 39 LỚP VÀ TỪ ĐIỂN PHÁC ĐỒ ĐIỀU TRỊ
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
    'ZZ_Khong_Phai_La_Cay'
]

# Từ điển ánh xạ (Mapping) bệnh với phác đồ điều trị
treatment_dict = {
    "Khỏe mạnh": "🌱 Cây đang phát triển tốt! Tiếp tục duy trì chế độ dinh dưỡng, tưới tiêu hợp lý và thường xuyên theo dõi để phòng bệnh.",
    "Bệnh sương mai": "💧 Cắt bỏ và tiêu hủy lá bệnh. Tránh tưới nước vào chiều tối. Phun thuốc trừ nấm chứa hoạt chất Mancozeb hoặc Metalaxyl.",
    "Bệnh phấn trắng": "🌬️ Đảm bảo vườn thông thoáng. Sử dụng thuốc trừ nấm gốc lưu huỳnh hoặc thuốc sinh học chiết xuất tỏi/neem.",
    "Bệnh gỉ sắt": "🍂 Thu bản lá rụng và tiêu hủy. Phun thuốc đặc trị nấm (Propiconazole, Tebuconazole). Bón thêm Kali để tăng sức đề kháng.",
    "Bệnh đốm vi khuẩn": "✂️ Vi khuẩn lây qua nước. Ngừng tưới phun sương lên mặt lá. Sử dụng các loại thuốc diệt khuẩn gốc Đồng (Copper-based).",
    "Bệnh nấm lá": "🍄 Tăng cường thoát nước, tỉa thưa cành lá. Phun thuốc nấm phổ rộng (Chlorothalonil) kết hợp bón phân hữu cơ.",
    "Nhện đỏ": "🕷️ Rửa lá bằng vòi xịt áp lực cao để đẩy nhện đi. Dùng dầu khoáng sinh học, hoặc thuốc trị nhện chuyên dụng (Abamectin).",
    "Bệnh xoăn vàng lá": "🦠 Bệnh do virus lây qua côn trùng (bọ phấn trắng). Lập tức nhổ bỏ và tiêu hủy cây bệnh để tránh lây lan. Xịt thuốc diệt côn trùng.",
    "Bệnh khảm": "🦠 Bệnh virus không có thuốc chữa. Rửa sạch nông cụ, nhổ bỏ cây bệnh. Phun thuốc kiểm soát rệp/bọ trĩ truyền bệnh.",
    "Bệnh vàng lá gân xanh": "⚠️ Bệnh Greening cực kỳ nguy hiểm. Buộc phải chặt bỏ cành/cây bệnh và tiêu hủy. Diệt rầy chổng cánh để phòng ngừa."
}

def get_treatment(disease_name):
    if "Khỏe mạnh" in disease_name: return treatment_dict["Khỏe mạnh"]
    for keyword, treatment in treatment_dict.items():
        if keyword.lower() in disease_name.lower(): return treatment
    return "Nông dân cần cách ly phần bị viêm nhiễm, đảm bảo thoát nước tốt và mang mẫu lá đến trạm khuyến nông gần nhất để được hỗ trợ chuyên sâu."

# ==========================================
# 3. LUỒNG XỬ LÝ: CHẨN ĐOÁN VÀ XUẤT BÁO CÁO
# ==========================================
def predict_and_report(img):
    if img is None:
        return {"Vui lòng cung cấp hình ảnh.": 1.0}, "Không có dữ liệu.", None

    try:
        # Tiền xử lý
        img_resized = img.convert("RGB").resize((224, 224))
        img_array = tf.expand_dims(tf.keras.utils.img_to_array(img_resized), 0)
        
        # Dự đoán
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        top_disease = class_names[predicted_idx]

        # Bộ lọc ảnh rác OOD
        if top_disease == 'ZZ_Khong_Phai_La_Cay':
            return {"⚠️ Lỗi: Không phải lá cây nông nghiệp": 1.0}, "Hình ảnh không hợp lệ, hệ thống từ chối đưa ra phác đồ.", None
        if predictions[predicted_idx] < 0.45:
            return {"⚠️ Cảnh báo: Ảnh quá nhiễu/mờ": 1.0}, "Hệ thống không đủ tự tin nhận diện. Vui lòng lấy nét và chụp lại ảnh.", None

        # Trích xuất kết quả
        confidences = {class_names[i]: float(predictions[i]) for i in range(38)}
        recommendation = get_treatment(top_disease)

        # Xuất file báo cáo
        report_filename = f"BaoCao_ChanDoan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("    BÁO CÁO CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG AI      \n")
            f.write("==================================================\n\n")
            f.write(f"Thời gian chẩn đoán: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            f.write("1. KẾT QUẢ PHÂN TÍCH:\n")
            f.write("-" * 30 + "\n")
            f.write(f" - Chẩn đoán chính: {top_disease}\n")
            f.write(f" - Độ tin cậy (AI Confidence): {predictions[predicted_idx]*100:.2f}%\n\n")
            f.write("2. KHUYẾN NGHỊ VÀ PHÁC ĐỒ ĐIỀU TRỊ:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{recommendation}\n\n")
            f.write("==================================================\n")
            f.write("Lưu ý: Báo cáo được tạo tự động từ hệ thống MobileNetV2.\n")
            f.write("Người thực hiện: Nguyễn Huỳnh Giao - Khoa CNTT, ĐH Nông Lâm.\n")

        return confidences, recommendation, report_filename

    except Exception as e:
        return {f"⚠️ Lỗi xử lý: {str(e)}": 1.0}, "Lỗi hệ thống.", None

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
    gr.Markdown("# 🌱 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG (MOBILE APP)")
    gr.Markdown("**Hướng dẫn:** Dùng điện thoại chụp mặt lá bị tổn thương. Hệ thống sẽ phân tích, lập phác đồ điều trị và xuất báo cáo kỹ thuật.")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                # CHỈNH SỬA Ở ĐÂY: Thêm mirror_webcam=False cho chuẩn điện thoại
                with gr.Tab("📷 Chụp bằng Điện thoại"):
                    cam_input = gr.Image(type="pil", sources=["webcam"], mirror_webcam=False, label="Camera sau")
                    cam_btn = gr.Button("🔍 Quét Camera & Lập Báo Cáo", variant="primary")
                    
                with gr.Tab("🖼️ Tải ảnh từ Thư viện"):
                    up_input = gr.Image(type="pil", sources=["upload"], label="Chọn ảnh")
                    up_btn = gr.Button("🔍 Phân Tích & Lập Báo Cáo", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=3, label="Biểu đồ Xác suất AI")
            treatment_output = gr.Textbox(label="📋 Phác đồ Điều trị & Khuyến nghị", lines=3, interactive=False)
            file_output = gr.File(label="📥 Tải Báo Cáo Kỹ Thuật (.TXT)")

    # Gắn sự kiện
    up_btn.click(fn=predict_and_report, inputs=up_input, outputs=[label_output, treatment_output, file_output])
    cam_btn.click(fn=predict_and_report, inputs=cam_input, outputs=[label_output, treatment_output, file_output])

# ==========================================
# 5. KHỞI CHẠY MÁY CHỦ
# ==========================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
