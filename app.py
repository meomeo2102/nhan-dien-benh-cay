import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Tải mô hình AI
print("Đang tải mô hình...")
model = tf.keras.models.load_model('best_plant_model.keras')

# 2. Danh sách 38 loại bệnh (Đã được Việt hóa và định dạng đẹp mắt)
class_names = [
    'Táo - Bệnh vảy đen (Apple scab)', 
    'Táo - Bệnh thối đen (Black rot)', 
    'Táo - Bệnh gỉ sắt (Cedar apple rust)', 
    'Táo - Khỏe mạnh (Healthy)',
    'Việt quất - Khỏe mạnh (Healthy)', 
    'Anh đào - Bệnh phấn trắng (Powdery mildew)', 
    'Anh đào - Khỏe mạnh (Healthy)',
    'Ngô - Bệnh đốm xám (Cercospora leaf spot)', 
    'Ngô - Bệnh gỉ sắt (Common rust)',
    'Ngô - Bệnh cháy lá (Northern Leaf Blight)', 
    'Ngô - Khỏe mạnh (Healthy)', 
    'Nho - Bệnh thối đen (Black rot)',
    'Nho - Bệnh sởi đen (Esca - Black Measles)', 
    'Nho - Bệnh đốm lá (Leaf blight)', 
    'Nho - Khỏe mạnh (Healthy)',
    'Cam/Chanh - Bệnh vàng lá gân xanh (Citrus greening)', 
    'Đào - Bệnh đốm vi khuẩn (Bacterial spot)', 
    'Đào - Khỏe mạnh (Healthy)',
    'Ớt chuông - Bệnh đốm vi khuẩn (Bacterial spot)', 
    'Ớt chuông - Khỏe mạnh (Healthy)', 
    'Khoai tây - Bệnh sương mai sớm (Early blight)',
    'Khoai tây - Bệnh sương mai muộn (Late blight)', 
    'Khoai tây - Khỏe mạnh (Healthy)', 
    'Mâm xôi - Khỏe mạnh (Healthy)', 
    'Đậu nành - Khỏe mạnh (Healthy)',
    'Bí đỏ - Bệnh phấn trắng (Powdery mildew)', 
    'Dâu tây - Bệnh cháy lá (Leaf scorch)', 
    'Dâu tây - Khỏe mạnh (Healthy)',
    'Cà chua - Bệnh đốm vi khuẩn (Bacterial spot)', 
    'Cà chua - Bệnh đốm vòng (Early blight)', 
    'Cà chua - Bệnh sương mai muộn (Late blight)', 
    'Cà chua - Bệnh nấm lá (Leaf Mold)',
    'Cà chua - Bệnh đốm lá Septoria (Septoria leaf spot)', 
    'Cà chua - Nhện đỏ (Spider mites)', 
    'Cà chua - Bệnh đốm đích (Target Spot)',
    'Cà chua - Bệnh xoăn vàng lá (Yellow Leaf Curl Virus)', 
    'Cà chua - Bệnh khảm (Mosaic virus)', 
    'Cà chua - Khỏe mạnh (Healthy)'
]

# 3. Hàm xử lý ảnh và dự đoán tích hợp cảnh báo
def predict_disease(img):
    if img is None:
        return "Vui lòng tải lên một hình ảnh."

    # Tiền xử lý ảnh
    img_resized = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    # Dự đoán
    predictions = model.predict(img_array)[0]
    
    # Lấy ra xác suất cao nhất
    max_confidence = np.max(predictions)
    
    # CẢNH BÁO: Nếu xác suất cao nhất mà dưới 45% -> Khả năng cao không phải là lá cây
    if max_confidence < 0.45:
        return {"⚠️ Lỗi: Hình ảnh không hợp lệ (Không phải lá cây nông nghiệp)": float(1.0)}

    # Trả về kết quả bình thường nếu xác suất cao
    confidences = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return confidences

# 4. CSS
custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #2e7d32; /* Màu xanh lá cây đậm */
        text-align: center;
    }
    p {
        text-align: center;
        font-size: 16px;
    }
    .footer {display: none !important} /* Ẩn footer của Gradio cho chuyên nghiệp */
"""

demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Tải ảnh hoặc chụp ảnh từ Camera"),
    outputs=gr.Label(num_top_classes=3, label="Phân tích và Chẩn đoán"),
    title="🌱 HỆ THỐNG TRÍ TUỆ NHÂN TẠO CHẨN ĐOÁN BỆNH CÂY TRỒNG",
    description="<b>Hướng dẫn:</b> Hãy tải lên hoặc dùng điện thoại chụp ảnh lá cây (Cà chua, Ngô, Khoai tây...). Hệ thống AI (MobileNetV2) sẽ phân tích tổn thương và trả về 3 kết quả có khả năng cao nhất.",
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"), 
    css=custom_css,
    allow_flagging="never" 
)

# 5. Khởi chạy
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
