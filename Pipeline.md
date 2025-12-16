## 1. Thu thập dữ liệu
- Dữ liệu được tải xuống từ kaggle:

```python
import kagglehub

path = kagglehub.dataset_download("guutran/vietnamese-sentiment-analysis-food-reviews")
print("Path to dataset files:", path)
```

- Quan sát, EDA phân phối các class, số token,...
- Dữ liệu được lưu tại thư mục Data

## 2. Tiền xử lý dữ liệu
- Chuẩn hóa văn bản như chuyển về chữ thường, xử lý các ký tự đặc biệt, từ kéo dài, emoji,...
- Chuẩn hóa teencode, viết tắt.
- Loại bỏ stopwords (optional do PhoBert thường không yêu cầu //cre ChatGPT chưa kiểm chứng legit)
- Tách từ cho các mô hình `SVM, RNN, CNN-LSTM, BiLSTM` bằng `ViTokenizer` của pyvi. Với `PhoBert` thì không cần do đã có BPE xử lý.

## 3. Chia dữ liệu
- Hiện tại dữ liệu đang có sẵn 2 tập train và test, tập test chưa có nhãn nên sẽ không sử dụng.
- Dùng train_test_split với random state là 2020, test size là 0.2, stratify = y để chia train và test từ file train và đảm bảo tỉ lệ nhãn.
- Với các mô hình học sâu, khi huấn luyện, dùng thêm validation split là 0.2.

## 4. Huấn luyện mô hình
### 4.1 SVM
- Dùng TF-IDF làm vectorizer
- Huấn luyện và tinh chỉnh siêu tham số của SVM.
### 4.2 RNN, CNN-LSTM, BiLSTM
- Dùng embedding có sẵn của fastText.
```python
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz
!gunzip cc.vi.300.vec.gz
```
- Huấn luyện và tự định nghĩa cấu trúc các mô hình.
### 4.3 PhoBert-base-v2
- Dữ liệu đầu vào không được tách từ sẵn, phải ở dạng nguyên câu, chúng sẽ được xử lý và tách từ bằng `RDRSegmenter` và mã hóa bằng `BPE`.
- Embedding sẽ được tạo trong chính mô hình bằng `BPE` và `Transformer`.
- Tại đây đang full finetune từ pretrained model.
- Tokenizer và thêm Classification Head để huấn luyện, dưới đây là code minh họa:

- Tạo encoder
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

def encode(texts):
    return tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )
```

- Thêm classification head
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base-v2",
    num_labels=2  # positive / negative
)
```

- Tạo dataloader
```python
import torch

class PhoDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = encode(texts)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
```

- Train và test
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./phobert-v2-sentiment',
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
pred = trainer.predict(test_dataset)
y_pred = pred.predictions.argmax(axis=1)
```

- Predict trên 1 mẫu
```python
def predict(text):
    inputs = encode([text])
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=1).item()
```

## 5. Đánh giá kết quả
- So sánh các mô hình trên độ đo về accuracy, f1-macro, precision, recall
- Nhận xét và đánh giá lỗi
- Phân tích kết quả với 1 vài mẫu

## 6. Demo (Nếu còn thời gian)
- Tạo ứng dụng gợi ý và đánh giá các quán ăn dựa trên dữ liệu được crawl về từ facebook, foody,...
- Có thể đề xuất quán ăn dựa trên tỉ lệ tích cực/ tiêu cực, tích cực/ tổng.
- Tự gán nhãn bình luận tích cực/tiêu cực.

## 7. Yêu cầu slide và báo cáo
- Nội dung, mô tả đề tài
- Thu thập dữ liệu
- Phân tích ngữ liệu, nhận xét
- Mô hình sử dụng (1 mô hình duy nhất, nắm rõ)
- Huấn luyện, đánh giá, so sánh
- Nhận xét lỗi, phân tích ví dụ, vì sao?