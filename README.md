Dịch vụ Xác thực Gương mặt (Face Verification Service)

Đây là một microservice chuyên dụng được xây dựng bằng Python và FastAPI, cung cấp chức năng nhận diện và xác thực khuôn mặt. Dịch vụ này được tích hợp vào hệ thống để phục vụ cho các tính năng như chấm công nhân viên bằng nhận diện khuôn mặt.

Chức Năng Chính

Đăng ký Gương mặt: Cung cấp một API endpoint để đăng ký và lưu trữ vector đặc trưng (embedding) của khuôn mặt một nhân viên mới.

Xác thực Gương mặt: Cung cấp một API endpoint để nhận diện và xác thực một hình ảnh khuôn mặt so với cơ sở dữ liệu đã lưu trữ, trả về thông tin của nhân viên tương ứng nếu tìm thấy.

Quản lý Cơ sở dữ liệu Gương mặt: Lưu trữ và quản lý các vector đặc trưng của khuôn mặt trong một file face_database.pkl để truy xuất nhanh chóng.

Công Nghệ Sử Dụng

Ngôn ngữ: Python

Framework: FastAPI

Thư viện Nhận diện Gương mặt: DeepFace

Thư viện Web Server: Uvicorn

Hướng Dẫn Cài Đặt và Chạy Dự Án

Yêu cầu tiên quyết

Python (phiên bản 3.9 trở lên)

pip (trình quản lý gói của Python)

Cài đặt

Clone repository:

git clone <your-repo-url>
cd verify-face


Tạo môi trường ảo (khuyến khích):

python -m venv venv
source venv/bin/activate  # Trên Linux/macOS
.\venv\Scripts\activate  # Trên Windows


Cài đặt các dependencies:

pip install -r requirements.txt


Chạy ứng dụng

Khởi động server:
Sử dụng Uvicorn để chạy ứng dụng FastAPI. File chính của ứng dụng là nhandien.py.

uvicorn nhandien:app --host 0.0.0.0 --port 8010 --reload


--host 0.0.0.0: Cho phép truy cập từ bên ngoài container/máy ảo.

--port 8010: Chạy dịch vụ ở cổng 8010.

--reload: Tự động khởi động lại server khi có thay đổi trong mã nguồn.

Chạy bằng Docker:
Dự án đã có sẵn Dockerfile và docker-compose.yml.

docker-compose up --build


API Endpoints

Sau khi khởi động, dịch vụ sẽ cung cấp các API sau:

POST /register-face/{id}:

Mô tả: Đăng ký khuôn mặt cho một nhân viên với id tương ứng.

Body: Gửi một file hình ảnh.

Phản hồi: Thông báo đăng ký thành công hoặc thất bại.

POST /verify-face:

Mô tả: Xác thực một khuôn mặt từ hình ảnh được gửi lên.

Body: Gửi một file hình ảnh.

Phản hồi: Trả về id của nhân viên được nhận diện hoặc thông báo không tìm thấy.

Tài liệu API chi tiết (do FastAPI tự động tạo) có thể được truy cập tại http://localhost:8010/docs.
