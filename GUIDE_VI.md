# Hướng dẫn làm project SMS Spam Classifier

Project nhận một SMS tiếng Anh và dự đoán `spam` hoặc `ham`. `Ham` là tên nhãn của tin nhắn bình thường. Bạn sẽ tải dữ liệu có nhãn, huấn luyện mô hình, đọc kết quả, chạy demo và đưa repo lên GitHub.

## 1. Mở project trên máy

Giải nén ZIP vào thư mục `github-portfolio` bạn đã dùng cho repo pandas. File ZIP đã chứa thư mục `sms-spam-classifier`. Mở PowerShell và chạy.

```powershell
cd "$env:USERPROFILE\github-portfolio\sms-spam-classifier"
dir
```

Bạn đang đúng thư mục khi thấy `app.py`, `train.py` và `requirements.txt`.

## 2. Cài thư viện trong môi trường riêng

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.venv` chứa Python và các thư viện riêng cho project. Đường dẫn `.\.venv\Scripts\python.exe` chạy đúng Python đó. File `requirements.txt` ghi các phiên bản thư viện đã dùng để kiểm tra project trên Windows với Python 3.13.

## 3. Tải dữ liệu và huấn luyện mô hình

```powershell
Invoke-WebRequest -UseBasicParsing -Uri 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip' -OutFile 'data\sms-spam-collection.zip'
.\.venv\Scripts\python.exe train.py
```

Nếu lệnh tải gặp lỗi, dùng nút **Download** trên trang UCI rồi lưu cả file ZIP vào thư mục `data` với tên `sms-spam-collection.zip`. Mỗi dòng dữ liệu có nhãn và nội dung SMS, cách nhau bằng dấu tab. [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Trong `train.py`, đọc từ hàm `main()` để thấy thứ tự xử lý.

1. **Làm sạch.** Chuẩn hóa khoảng trắng, loại tin nhắn trùng và bỏ các tin trùng nội dung có nhãn mâu thuẫn. Làm việc này trước khi chia tập để một bản sao y hệt không lọt vào cả hai tập.
2. **Chia dữ liệu.** Khoảng 80% dùng để học, 20% giữ riêng để đo. Chia theo kiểu *stratified* để tỷ lệ spam trong hai tập gần nhau.
3. **Tạo mốc so sánh.** Baseline luôn đoán lớp phổ biến nhất là `ham`. Nó cho biết kết quả khi chưa học gì từ nội dung tin nhắn.
4. **Huấn luyện.** TF-IDF chuyển chữ thành số dựa trên độ phổ biến của từ và cụm hai từ. Logistic regression học từ các số đó để phân biệt hai nhãn. `Pipeline` nối hai bước, giúp TF-IDF chỉ học từ tập train và dùng cùng cách xử lý khi dự đoán.
5. **Đánh giá và lưu.** Script ghi mô hình vào `artifacts/sms-pipeline.joblib` và số liệu vào `reports/metrics.json`.

## 4. Đọc kết quả và mở demo

Kết quả tham chiếu dùng 1.032 tin nhắn kiểm thử, trong đó có 128 spam.

| Số đo | Ý nghĩa | Kết quả mô hình |
| --- | --- | ---: |
| Accuracy | Tỷ lệ đoán đúng trên toàn bộ tin nhắn | 98,16% |
| Spam precision | Trong các tin bị gắn spam, bao nhiêu tin đúng là spam | 90,98% |
| Spam recall | Trong các tin spam thật, mô hình tìm được bao nhiêu | 94,53% |
| Spam F1 | Một số đo kết hợp precision và recall, từ 0 tới 1 | 0,9272 |

Baseline đạt accuracy 87,60% dù bỏ sót hết 128 tin spam. Mô hình bắt được 121 tin spam, bỏ sót 7 tin và gắn nhầm 12 tin thường. Tập dữ liệu là SMS tiếng Anh thu thập từ lâu. Kết quả có thể khác với SMS tiếng Việt hoặc dữ liệu hiện nay.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.headless true
```

Hai kiểm thử kiểm tra cách loại bản trùng và cách tính số đo cho lớp spam. Khi server chạy, mở [http://127.0.0.1:8501](http://127.0.0.1:8501). Chọn **Try a normal SMS** hoặc **Try a spam SMS**, rồi bấm **Classify message**. Bạn cũng có thể nhập tin nhắn tự nghĩ. Quay lại PowerShell và nhấn **Ctrl+C** để dừng demo.

## 5. Đăng repo lên GitHub

Chạy từng dòng từ thư mục `sms-spam-classifier`. Nếu gặp lỗi, dừng ở lệnh đang lỗi. `git status --short` cho biết các file chuẩn bị commit. Kiểm tra danh sách có README, mã Python, test và báo cáo. `.gitignore` giữ `.venv`, bộ dữ liệu và mô hình đã huấn luyện ở máy bạn.

```powershell
git init -b main
git add .
git status --short
git commit -m "Add SMS spam classifier and evaluation"
gh repo create Hoangchuyenanh/sms-spam-classifier --public --description "English SMS spam classifier with scikit-learn and a Streamlit demo"
git remote add origin https://github.com/Hoangchuyenanh/sms-spam-classifier.git
git push -u origin main
```

`gh repo create` tạo repo trên GitHub. `git remote add` lưu địa chỉ của repo dưới tên `origin`. `git push -u` gửi commit và liên kết nhánh `main` với nhánh trực tuyến để lần sau chỉ cần `git push`. [GitHub CLI](https://cli.github.com/manual/gh_repo_create) · [Hướng dẫn push code từ máy](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

Mở repo vừa tạo và kiểm tra README tiếng Anh, bảng kết quả và link đến báo cáo. Sau đó vào **Your profile → Customize your pins**, chọn `sms-spam-classifier` và lưu. [Ghim repo trên GitHub](https://docs.github.com/en/account-and-profile/how-tos/profile-customization/pinning-items-to-your-profile)

## 6. Thêm đóng góp của bạn bằng một pull request

Tự nghĩ năm tin nhắn tiếng Anh. Ghi nhãn bạn dự đoán trước, chạy demo rồi ghi nhãn mô hình trả về. Nếu gặp trường hợp đoán sai, viết một câu nhận xét bằng tiếng Anh. Đây là kiểm tra bổ sung trên ví dụ của bạn và bảng kết quả UCI vẫn giữ số liệu từ tập test cố định.

```powershell
git switch -c docs/check-sms-examples
notepad reports\example-checks.md
```

Lưu bảng với các cột **Example**, **Expected label**, **Model prediction** và **Notes**. Sau khi đóng Notepad, chạy tiếp.

```powershell
git add reports/example-checks.md
git commit -m "Document predictions on five synthetic SMS examples"
git push -u origin docs/check-sms-examples
gh pr create --base main --head docs/check-sms-examples --title "Document manual SMS checks" --body "Compare expected and predicted labels for five synthetic messages."
```

Mở link pull request mà GitHub CLI trả về. Xem **Files changed**, rồi dùng **Merge pull request** nếu nội dung đúng. Để cập nhật `main` trên máy sau khi merge, chạy `git switch main` rồi `git pull --ff-only`. [Tạo pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
