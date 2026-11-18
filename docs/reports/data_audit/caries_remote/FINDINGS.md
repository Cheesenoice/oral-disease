# Kiểm tra sơ bộ D1 — 20/10/2026

Đọc central directory và một số annotation/ảnh trực tiếp từ ZIP nguồn qua HTTP Range; **chưa thay thế kiểm tra MD5 của toàn archive**.

| Phát hiện thật | Bằng chứng / xử lý |
|---|---|
| ZIP chứa 6.265 JPG, 2.245 TXT và 2.245 XML | `paths.txt`, `preliminary_report.json`. Con số ảnh khác 6.313 trong mô tả; không tự cộng thêm ảnh từ benchmark archive |
| Nhiều ảnh không có file YOLO tương ứng trong archive | Chưa kết luận là ảnh âm tính; cần đối chiếu COCO/LabelMe và mô tả nguồn đầy đủ |
| Mẫu đầu tiên có box vượt ảnh | File `anonymous_003-007-1214-00_1732862917124_Frontal_View`: YOLO `cy=1.746...`; XML y=1419..1685 nhưng chiều cao ảnh thực tế 889 |
| Vượt ảnh không được giải thích bằng EXIF ở mẫu đó | Ảnh thực tế và XML đều 1583×889, không có tag orientation. Không tự xoay hoặc cắt box để “sửa” |
| Hai mẫu tiếp theo có box nhìn thấy trên ảnh | `preliminary_overlays.jpg`; kiểm tra rộng hơn trước khi duyệt mapping |
| Mẫu đối chiếu lớp | Trong ba mẫu: YOLO 0 tương ứng XML `D`, YOLO 1 tương ứng `d`. Chưa coi ba mẫu là kiểm chứng toàn bộ mã lớp |

**Quyết định:** vẫn giữ D1 làm nguồn caries. Audit toàn bộ sau khi tải xong, cách ly annotation không hợp lệ có lý do rõ ràng; không train bằng nhãn chưa kiểm tra. Nếu phải loại mẫu, công bố số lượng trước/sau làm sạch và danh sách lý do.
