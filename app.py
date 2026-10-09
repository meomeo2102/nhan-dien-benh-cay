import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Tải mô hình AI
model = tf.keras.models.load_model('best_plant_model.keras')

class_names = [
    'Táo - Bệnh vảy đen (Apple scab)', 'Táo - Bệnh thối đen (Black rot)', 'Táo - Bệnh gỉ sắt (Cedar apple rust)', 'Táo - Khỏe mạnh (Healthy)',
    'Việt quất - Khỏe mạnh (Healthy)', 'Anh đào - Bệnh phấn trắng (Powdery mildew)', 'Anh đào - Khỏe mạnh (Healthy)',
    'Ngô - Bệnh đốm xám (Cercospora leaf spot)', 'Ngô - Bệnh gỉ sắt (Common rust)', 'Ngô - Bệnh cháy lá (Northern Leaf Blight)', 'Ngô - Khỏe mạnh (Healthy)', 
    'Nho - Bệnh thối đen (Black rot)', 'Nho - Bệnh sởi đen (Esca - Black Measles)', 'Nho - Bệnh đốm lá (Leaf blight)', 'Nho - Khỏe mạnh (Healthy)',
    'Cam/Chanh - Bệnh vàng lá gân xanh (Citrus greening)', 'Đào - Bệnh đốm vi khuẩn (Bacterial spot)', 'Đào - Khỏe mạnh (Healthy)',
    'Ớt chuông - Bệnh đốm vi khuẩn (Bacterial spot)', 'Ớt chuông - Khỏe mạnh (Healthy)', 'Khoai tây - Bệnh sương mai sớm (Early blight)',
    'Khoai tây - Bệnh sương mai muộn (Late blight)', 'Khoai tây - Khỏe mạnh (Healthy)', 'Mâm xôi - Khỏe mạnh (Healthy)', 
    'Đậu nành - Khỏe mạnh (Healthy)', 'Bí đỏ - Bệnh phấn trắng (Powdery mildew)', 'Dâu tây - Bệnh cháy lá (Leaf scorch)', 'Dâu tây - Khỏe mạnh (Healthy)',
    'Cà chua - Bệnh đốm vi khuẩn (Bacterial spot)', 'Cà chua - Bệnh đốm vòng (Early blight)', 'Cà chua - Bệnh sương mai muộn (Late blight)', 
    'Cà chua - Bệnh nấm lá (Leaf Mold)', 'Cà chua - Bệnh đốm lá Septoria (Septoria leaf spot)', 'Cà chua - Nhện đỏ (Spider mites)', 'Cà chua - Bệnh đốm đích (Target Spot)',
    'Cà chua - Bệnh xoăn vàng lá (Yellow Leaf Curl Virus)', 'Cà chua - Bệnh khảm (Mosaic virus)', 'Cà chua - Khỏe mạnh (Healthy)'
]

def predict_disease(img):
    if img is None:
        return "Vui lòng tải lên một hình ảnh."

    img_resized = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)[0]
    
    # Lọc kép: Xác suất cao nhất và độ chênh lệch
    sorted_preds = np.sort(predictions)[::-1]
    top_1_confidence = sorted_preds[0]
    top_2_confidence = sorted_preds[1]
    
    if top_1_confidence < 0.60 or (top_1_confidence - top_2_confidence) < 0.20:
        return {"⚠️ Lỗi: Hình ảnh không hợp lệ hoặc không phải lá cây nông nghiệp. Vui lòng thử lại!": 1.0}

    confidences = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return confidences

custom_css = """
    .gradio-container { font-family: 'Arial', sans-serif; }
    h1 { color: #2e7d32; text-align: center; }
    p { text-align: center; font-size: 16px; }
    .footer {display: none !important} 
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate")) as demo:
    gr.Markdown("# 🌱 HỆ THỐNG TRÍ TUỆ NHÂN TẠO CHẨN ĐOÁN BỆNH CÂY TRỒNG")
    gr.Markdown("**Hướng dẫn:** Tải lên ảnh có sẵn hoặc dùng Camera/Webcam chụp lá cây. Hệ thống AI (MobileNetV2) sẽ phân tích tổn thương và trả về 3 kết quả có khả năng cao nhất.")
    
    with gr.Row():
        image_input = gr.Image(type="pil", sources=["upload", "webcam"], interactive=True, label="Khu vực nhập hình ảnh")
    
    with gr.Row():
        predict_btn = gr.Button("🔍 Tiến hành Chẩn Đoán", variant="primary")
        
    with gr.Row():
        label_output = gr.Label(num_top_classes=3, label="Kết quả phân tích")
        
    # Gắn sự kiện click
    predict_btn.click(fn=predict_disease, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
