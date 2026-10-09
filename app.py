import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Tải mô hình AI (file phải nằm cùng thư mục với app.py)
print("Đang tải mô hình...")
model = tf.keras.models.load_model('best_plant_model.keras')

# 2. Danh sách 38 loại bệnh
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# 3. Hàm xử lý ảnh và dự đoán
def predict_disease(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)[0]
    confidences = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return confidences

# 4. Cấu hình giao diện Gradio
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🌱 Trợ lý AI Chẩn Đoán Bệnh Cây Trồng",
    description="Chụp hoặc tải lên hình ảnh lá cây bị bệnh để AI phân tích và chẩn đoán.",
    theme="default"
)

# 5. Khởi chạy
if __name__ == "__main__":
    demo.launch(share=True) # share=True để tạo link chia sẻ public