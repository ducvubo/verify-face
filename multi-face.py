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
DATABASE_FILE = "/app/data/face_database-multi.pkl"

# Khởi tạo InsightFace
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # Thay bằng "CUDAExecutionProvider" nếu dùng GPU
face_app.prepare(ctx_id=0)

# Tải cơ sở dữ liệu khuôn mặt
def load_database():
    if os.path.exists(DATABASE_FILE) and os.path.getsize(DATABASE_FILE) > 0:
        try:
            with open(DATABASE_FILE, "rb") as f:
                database = pickle.load(f)
            # Chuyển đổi dữ liệu cũ (nếu có) sang định dạng mới
            for name in database:
                if isinstance(database[name], np.ndarray):
                    database[name] = [database[name]]  # Chuyển ndarray thành danh sách
            return database
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
    try:
        with open(DATABASE_FILE, "wb") as f:
            pickle.dump(database, f)
        print(f"Đã lưu cơ sở dữ liệu vào {DATABASE_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu cơ sở dữ liệu: {e}")

# Thêm người vào cơ sở dữ liệu
@app.route('/add_person', methods=['POST'])
def add_person():
    person_name = request.form.get("name")
    files = request.files.getlist("images")

    if not person_name or not files:
        return jsonify({"error": "Tên và hình ảnh là bắt buộc"}), 400

    # Tải cơ sở dữ liệu
    database = load_database()

    # Khởi tạo danh sách embedding cho person_name nếu chưa tồn tại
    if person_name not in database:
        database[person_name] = []
    elif not isinstance(database[person_name], list):
        # Chuyển đổi dữ liệu cũ (nếu có) thành danh sách
        database[person_name] = [database[person_name]]

    # Lưu hình ảnh tạm thời
    temp_dir = f"temp_images_{person_name}"
    os.makedirs(temp_dir, exist_ok=True)
    image_paths = []
    for file in files:
        img_path = os.path.join(temp_dir, file.filename)
        file.save(img_path)
        image_paths.append(img_path)

    # Mã hóa khuôn mặt từ tất cả ảnh
    initial_embedding_count = len(database[person_name])
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            faces = face_app.get(image_np)
            if faces:
                # Thêm embedding của khuôn mặt đầu tiên vào danh sách
                database[person_name].append(faces[0].embedding)
                print(f"Đã thêm embedding cho {person_name} từ {img_path}")
            else:
                print(f"Không phát hiện được khuôn mặt trong {img_path}")
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {e}")

    # Kiểm tra xem có embedding nào được thêm không
    if len(database[person_name]) == initial_embedding_count:
        shutil.rmtree(temp_dir)
        if not database[person_name]:
            del database[person_name]
        return jsonify({"error": "Không phát hiện được khuôn mặt trong bất kỳ hình ảnh nào"}), 400

    # Lưu cơ sở dữ liệu
    save_database(database)

    # Xóa tệp tạm thời
    shutil.rmtree(temp_dir)

    return jsonify({"message": f"Đã thêm {person_name} với {len(database[person_name])} khuôn mặt vào cơ sở dữ liệu"}), 200

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

@app.route('/list_people', methods=['GET'])
def list_people():
    database = load_database()
    names = list(database.keys())  # Lấy danh sách tên
    return jsonify(names), 200

# Liệt kê tất cả người có trong cơ sở dữ liệu
@app.route('/list_people_duc', methods=['GET'])
def list_people_duc():
    database = load_database()
    people = {name: len(embeddings) for name, embeddings in database.items()}
    return jsonify({"people": people}), 200

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
    threshold = 0.55
    attendance = []

    # So khớp từng khuôn mặt
    for face in faces:
        face_encoding = face.embedding
        for name, saved_embeddings in database.items():
            for saved_encoding in saved_embeddings:
                # Tính độ tương đồng cosine
                similarity = 1 - cosine(saved_encoding, face_encoding)
                print(f"So sánh với {name}: similarity = {similarity:.4f}")
                if similarity > threshold:
                    # Chuyển similarity thành float và làm tròn
                    confidence = float(similarity)
                    attendance.append({"name": name, "confidence": round(confidence, 2)})
                    break  # Thoát vòng lặp saved_embeddings nếu tìm thấy khớp

    os.remove(img_path)

    if attendance:
        return jsonify({"attendance": attendance}), 200
    return jsonify({"message": "Không tìm thấy người phù hợp"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)