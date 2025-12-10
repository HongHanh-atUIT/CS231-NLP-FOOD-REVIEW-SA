## Mô tả file trong ./Dataset
- `all-emoji.json`: dictionary chứa các emoji thông thường, có id, utf code, emoji và tên tương ứng.
- full-emoji-modifiers.json: dictionary chứa các emoji khác nhau về màu sắc, có id, utf code, emoji và tên tương ứng.
- emoji_dict.txt: file txt chứa ánh xạ từ cột 1 (emoji) sang tên được encode dạng emj_name ở cột 2. các emoji cùng flag sẽ cùng tên. Ví dụ cùng thuộc nhóm flag như cờ VN, cờ Mỹ, cờ TQ đều là emj_flag, hoặc nếu là vẫy tay màu da, màu vàng, màu nâu sẽ đều là emj_waving_hand.
- teencode.txt: file txt chứa ánh xạ từ cột 1 (teencode) sang từ chuẩn hóa của nó ở cột 2, lấy trên wiki, có thêm 1 số từ mới, có trùng lặp.
- unit_dict.txt: file txt chứa ánh xạ từ cột 1 (đơn vị) sang từ chuẩn hóa của nó ở cột 2, tự cài đặt.
- các file {...}clean.csv: file kết quả sau làm sạch.

## Mô tả các file code
- create_emoji_dict.ipynb: Biến đổi từ file json sang file txt duy nhất cho emoji
- EDA.ipynb: gồm EDA trước và sau khi tiền xử lý
- clean_dataset.ipynb: src chính tiền xử lý.

## Mô tả quá trình tiền xử lý
- Loại bỏ nan trong comment và rating
- Thay dấu xuống dòng bằng dấu cách
- Chuyển tất cả sang chữ thường
- Thêm khoảng cách sau dấu .
- Bỏ khoảng cách dư thừa
- Chuẩn hóa số-đơn vị như 1k là 1 ngàn, 1m là 1 mét,...
- Thêm khoảng cách trước và sau dấu câu rời để đảm bảo tách được từng từ, các dấu câu liền như !!! .. )) không được thêm khoảng cách.
- Tách từ bằng khoảng trắng
- Bỏ phần ký tự kéo dài
- Thay thế teencode và emoji được định nghĩa trong các dictionary.

## Vấn đề
- Tạo ra các từ là các ký tự đặc biệt, dấu câu, khiến số lượng của chúng quá nhiều, có thể cân nhắc loại bỏ but hong bit làm.

## Sử dụng
- Đối với các mô hình SVM và RNN base: load csv lên và tách bằng split, không dùng tokenizer có sẵn do dữ liệu bình luận không được formal, việc dùng tokenizer của underthesea là không cần thiết và dẫn tới tách từ sai.
- Đối với PhoBert: load csv lên và dùng trực tiếp, không cần tách thành các token.