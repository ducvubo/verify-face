from flask import Flask, jsonify, request
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import os
import pickle
import shutil
from scipy.spatial.distance import cosine

app = Flask(__name__)

# File lưu trữ cơ sở dữ liệu khuôn mặt
DATABASE_FILE = "face_database.pkl"

# Khởi tạo InsightFace
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # Thay bằng "CUDAExecutionProvider" nếu dùng GPU
face_app.prepare(ctx_id=0)

# Tải cơ sở dữ liệu khuôn mặt
def load_database():
    if os.path.exists(DATABASE_FILE) and os.path.getsize(DATABASE_FILE) > 0:
        try:
            with open(DATABASE_FILE, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Lỗi khi đọc file {DATABASE_FILE}: {e}. Tạo file mới.")
    else:
        print(f"File {DATABASE_FILE} không tồn tại hoặc rỗng. Tạo file mới.")
    
    # Tạo file mới với dictionary rỗng
    try:
        with open(DATABASE_FILE, "wb") as f:
            pickle.dump({}, f)
        print(f"Đã tạo file {DATABASE_FILE} với cơ sở dữ liệu rỗng.")
    except Exception as e:
        print(f"Lỗi khi tạo file {DATABASE_FILE}: {e}")
    
    return {}

# Lưu cơ sở dữ liệu khuôn mặt
def save_database(database):
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(database, f)

# Thêm người vào cơ sở dữ liệu
@app.route('/add_person', methods=['POST'])
def add_person():
    person_name = request.form.get("name")
    files = request.files.getlist("images")

    if not person_name or not files:
        return jsonify({"error": "Tên và hình ảnh là bắt buộc"}), 400

    # Tải cơ sở dữ liệu
    database = load_database()

    # Lưu hình ảnh tạm thời
    temp_dir = f"temp_images_{person_name}"
    os.makedirs(temp_dir, exist_ok=True)
    image_paths = []
    for file in files:
        img_path = os.path.join(temp_dir, file.filename)
        file.save(img_path)
        image_paths.append(img_path)

    # Mã hóa khuôn mặt
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        faces = face_app.get(image_np)
        if faces:
            # Chỉ lấy khuôn mặt đầu tiên
            database[person_name] = faces[0].embedding
        else:
            shutil.rmtree(temp_dir)
            return jsonify({"error": f"Không phát hiện được khuôn mặt trong hình: {img_path}"}), 400

    # Lưu cơ sở dữ liệu
    save_database(database)

    # Xóa tệp tạm thời
    shutil.rmtree(temp_dir)

    return jsonify({"message": f"Đã thêm {person_name} vào cơ sở dữ liệu"}), 200

# Xóa một người khỏi cơ sở dữ liệu
@app.route('/delete_person', methods=['DELETE'])
def delete_person():
    person_name = request.json.get("name")
    if not person_name:
        return jsonify({"error": "Tên là bắt buộc"}), 400

    # Tải cơ sở dữ liệu
    database = load_database()

    # Kiểm tra và xóa người
    if person_name in database:
        del database[person_name]
        save_database(database)
        return jsonify({"message": f"Đã xóa {person_name} khỏi cơ sở dữ liệu"}), 200
    else:
        return jsonify({"error": f"{person_name} không tồn tại trong cơ sở dữ liệu"}), 404

# Liệt kê tất cả người có trong cơ sở dữ liệu
@app.route('/list_people', methods=['GET'])
def list_people():
    database = load_database()
    return jsonify(list(database.keys())), 200

# Đánh dấu điểm danh
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Hình ảnh là bắt buộc"}), 400

    # Lưu hình ảnh tạm thời
    img_path = "attendance_image.jpg"
    file.save(img_path)

    # Tải và xử lý ảnh
    try:
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        os.remove(img_path)
        return jsonify({"error": f"Lỗi khi xử lý ảnh: {str(e)}"}), 400

    # Phát hiện khuôn mặt
    faces = face_app.get(image_np)
    print(f"Số khuôn mặt phát hiện: {len(faces)}")

    if not faces:
        os.remove(img_path)
        return jsonify({"error": "Không phát hiện được khuôn mặt trong hình"}), 400

    # Tải cơ sở dữ liệu
    database = load_database()
    print(f"Cơ sở dữ liệu: {list(database.keys())}")

    # Ngưỡng tương đồng
    threshold = 0.55  # Giảm ngưỡng để tăng khả năng khớp
    attendance = []

    # So khớp từng khuôn mặt
    for face in faces:
        face_encoding = face.embedding
        for name, saved_encoding in database.items():
            # Tính độ tương đồng cosine
            similarity = 1 - cosine(saved_encoding, face_encoding)
            print(f"So sánh với {name}: similarity = {similarity:.4f}")
            if similarity > threshold:
                # Chuyển similarity thành float và làm tròn
                confidence = float(similarity)
                attendance.append({"name": name, "confidence": round(confidence, 2)})

    os.remove(img_path)

    if attendance:
        return jsonify({"attendance": attendance}), 200
    return jsonify({"message": "Không tìm thấy người phù hợp"}), 404
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)