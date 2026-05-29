Dưới đây là nội dung chi tiết cho tệp `transunet_upgrade.md`. Bạn có thể sao chép toàn bộ phần văn bản này và lưu trực tiếp vào dự án của mình. Nội dung được thiết kế để giải thích rõ lý thuyết, kiến trúc, và các bước tích hợp mã nguồn cụ thể.

---

# Hướng dẫn Nâng cấp Mô hình Phân vùng Ảnh Y tế với TransUNet

Tài liệu này trình bày lộ trình và hướng dẫn kỹ thuật để nâng cấp hệ thống phân vùng ảnh tuyến vú (tập `Dataset_BUSI_with_GT`) từ U-Net thuần túy sang **TransUNet**.

Việc nâng cấp này giải quyết điểm yếu chí mạng của mạng CNN truyền thống: khả năng bao quát không gian (global context). Quá trình trích xuất đặc trưng toàn cục với Vision Transformer (ViT) chắc hẳn đã khá quen thuộc với môi trường dev local của bạn, và khi kết hợp bộ giải mã của U-Net, chúng ta sẽ có một mô hình tận dụng được cả thông tin tổng thể lẫn chi tiết cục bộ.

## 1. Tại sao lại là TransUNet?

Mô hình TransUNet là sự kết hợp (hybrid) giữa **CNN** và **Vision Transformer (ViT)**. Đối với ảnh siêu âm tuyến vú (có nhiều nhiễu, khối u không rõ viền, vị trí khối u thay đổi mạnh), TransUNet mang lại các lợi thế sau:

* **Bộ mã hóa ViT (Encoder):** Chia hình ảnh thành các patch nhỏ (ví dụ 16x16), sau đó tính toán sự phụ thuộc giữa tất cả các patch này thông qua cơ chế Self-Attention.
* **Bộ giải mã U-Net (Decoder):** Khôi phục lại độ phân giải ban đầu từ các vector đặc trưng của ViT, kết hợp với các skip-connections từ các lớp CNN trung gian để giữ lại độ sắc nét cho đường viền khối u.

Cơ chế cốt lõi của ViT trong TransUNet dựa trên phép toán Self-Attention, giúp mô hình hiểu được ngữ cảnh toàn bức ảnh ngay từ giai đoạn mã hóa:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## 2. Chuẩn bị Môi trường làm việc

Trước khi thay đổi cấu trúc cốt lõi trong `source/model.py`, hãy đảm bảo luồng công việc an toàn trên Git. Việc chỉnh sửa và thử nghiệm nên được thực hiện trên branch cá nhân của bạn:

```bash
# Đảm bảo bạn đang ở branch làm việc cá nhân
git checkout Nguyen_Quoc_Duy

# Cài đặt thêm thư viện (nếu cần) để hỗ trợ các thao tác tensor phức tạp của ViT
pip install einops

```

## 3. Cấu trúc Kiến trúc TransUNet

Kiến trúc sẽ thay đổi đáng kể ở phần **Encoder**. Thay vì dùng 4 khối `DoubleConv` và `MaxPool2d` liên tiếp, chúng ta thiết lập luồng đi như sau:

1. **CNN Feature Extractor (Tùy chọn nhưng khuyến nghị):** Ảnh đầu vào `256x256` đi qua một vài lớp ResNet cơ bản để lấy ra feature map trung gian (ví dụ `64x64`).
2. **Patch Embedding:** Cắt feature map thành các patch (ví dụ kích thước patch `P=16`), trải phẳng (flatten) và chiếu (project) thành các vector D-chiều.
3. **Transformer Encoder:** Đưa các patch embedding qua L lớp Transformer Blocks.
4. **Reshape:** Biến đổi chuỗi chuỗi vector đầu ra của Transformer trở lại không gian 2D.
5. **Decoder (Cascaded Upsampler):** Dùng `ConvTranspose2d` để upsample từng bước, ghép nối (concatenate) với các feature map lấy từ CNN Feature Extractor (Skip-connections).

## 4. Cập nhật mã nguồn (`source/model.py`)

Dưới đây là một bộ khung mã nguồn (skeleton) tối giản để bạn tích hợp TransUNet vào file `model.py` hiện tại.

```python
import torch
import torch.nn as nn
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim)
        )

    def forward(self, x):
        # x: (Batch, Num_Patches, Dim)
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x

class TransUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, out_channels=1, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 1. Patch Embedding (Sử dụng Conv2d với stride = patch_size để chia patch nhanh)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 2. Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 3. U-Net Decoder (Khôi phục độ phân giải)
        # Giả sử kích thước sau khi reshape là 16x16 (với ảnh 256x256, patch 16)
        self.up1 = nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2) # 16x16 -> 32x32
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)       # 32x32 -> 64x64
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)        # 64x64 -> 128x128
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)         # 128x128 -> 256x256
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # 4. Output Layer
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # --- ENCODER ---
        B, C, H, W = x.shape
        x = self.patch_embed(x)                     # (B, embed_dim, 16, 16)
        x = rearrange(x, 'b c h w -> b (h w) c')    # (B, 256, embed_dim)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # --- RESHAPE ---
        h = w = self.img_size // self.patch_size
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w) # (B, embed_dim, 16, 16)
        
        # --- DECODER (Chưa có skip connections để đơn giản hóa bản demo) ---
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        x = self.conv3(self.up3(x))
        x = self.conv4(self.up4(x))
        
        logits = self.out_conv(x)
        return logits

```

> **Lưu ý quan trọng:** Bản demo trên chưa bao gồm các **Skip-Connections** từ một mạng ResNet backbone. Trong mô hình TransUNet gốc, người ta thường dùng một ResNet50 ở giai đoạn đầu để trích xuất các feature map trung gian, sau đó mới đẩy vào ViT, rồi lấy các feature map đó nối (concat) vào quá trình Upsample ở Decoder. Bạn có thể tự mở rộng mã nguồn bằng cách thêm nhánh CNN song song.

## 5. Tinh chỉnh quá trình huấn luyện (`source/main.py`)

ViT nổi tiếng là "khát" dữ liệu (data-hungry) và thiếu quy nạp cục bộ (inductive bias). Do tập `Dataset_BUSI_with_GT` khá nhỏ, bạn cần thực hiện các điều chỉnh sau trong quá trình huấn luyện:

1. **Chỉnh sửa Optimizer:** Thay vì dùng Adam thông thường với `lr=0.001`, mô hình Transformer thường hội tụ tốt hơn với **AdamW** (Adam with Weight Decay) và tốc độ học nhỏ hơn (ví dụ: `1e-4` hoặc `3e-4`).
2. **Sử dụng Pre-trained Weights:** Huấn luyện ViT từ đầu (from scratch) trên vài nghìn ảnh siêu âm sẽ dẫn đến overfitting cực kỳ nhanh. Khuyến nghị lớn nhất là tải trọng số ViT đã được pre-train trên ImageNet (ví dụ `vit_base_patch16_224`) và chỉ tinh chỉnh (fine-tune) trên tập BUSI.
3. **Tăng cường số lượng Epochs:** Do tính chất của Attention, mô hình sẽ cần nhiều hơn 10 epochs để hội tụ hoàn toàn. Hãy xem xét tăng lên `50-100 epochs` kết hợp với Early Stopping.