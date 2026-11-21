&lt;p align=&quot;center&quot;&gt;
  &lt;a href=&quot;./README.md&quot;&gt;🇺🇸 English&lt;/a&gt; | &lt;strong&gt;🇻🇳 Tiếng Việt&lt;/strong&gt;
&lt;/p&gt;

# Máy Chủ AI Ternary-Bonsai-2-27B Trên Cloud (Kaggle & Google Colab)

Triển khai **Ternary-Bonsai-2-27B** (`PQ2_0-MTP` + Vision) thành API chuẩn OpenAI riêng tư trên GPU miễn phí (**Kaggle T4 x2** hoặc **Google Colab T4 / L4 / A100**) bằng `llama-server` bản CUDA 12.8 của PrismML.

Kho mã nguồn này tách quy trình thành **2 notebook riêng biệt** để bạn tốn **0 giờ GPU** cho việc tải mô hình nặng, khởi chạy máy chủ GPU chỉ trong **~30 giây**, tích hợp sẵn **bảo mật API Key**, **tùy chọn Ngrok hoặc Cloudflare Quick Tunnel**, và **tự động giữ kết nối (Keep-Alive)**.

---

## Cấu trúc kho mã nguồn

| Tệp | Phần cứng | Công dụng |
| :--- | :--- | :--- |
| `notebook_1_cpu_prep_keepalive_en.ipynb` | **Chỉ dùng CPU (0 giờ GPU)** | Tải sẵn binary `llama-server` CUDA 12.8 của PrismML, mô hình `Ternary-Bonsai-2-27B` GGUF (7.66 GB) và bộ chiếu Vision (`mmproj-Q8_0.gguf`, 0.63 GB) vào Kaggle Dataset hoặc Google Drive để tái sử dụng. |
| `notebook_2_gpu_server_keepalive_en.ipynb` | **GPU (T4 x2 / T4 / L4 / A100)** | Nạp nhanh toàn bộ tệp đã tải trong ~30 giây, chạy `llama-server` có khóa `--api-key`, mở đường hầm công khai qua **Ngrok** hoặc **Cloudflare Quick Tunnel**, và tự duy trì phiên chạy bằng vòng lặp Keep-Alive. |

---

## Tính năng nổi bật

- **Không lãng phí giờ GPU khi tải mô hình**: Tải một lần duy nhất bằng CPU ở Notebook 1 và gắn vào dùng ngay ở các phiên sau.
- **Bảo mật bằng `API_KEY` (`--api-key`)**: Khóa toàn bộ các endpoint `/v1/chat/completions`, `/v1/models` bằng Bearer token (trả về `HTTP 401` nếu thiếu hoặc sai khóa). Nếu bạn bỏ trống `API_KEY` khi `REQUIRE_API_KEY = True`, hệ thống tự sinh khóa ngẫu nhiên `sk-bonsai-...` và tự kiểm tra xác thực ngay khi khởi động.
- **Hỗ trợ song song Ngrok & Cloudflare Quick Tunnel**: Mặc định `TUNNEL_PROVIDER = "auto"` sẽ dùng **Ngrok** nếu bạn đã cấu hình `NGROK_AUTHTOKEN`, còn nếu không điền Secrets Ngrok sẽ tự động chuyển sang **Cloudflare Quick Tunnel** (`https://<random>.trycloudflare.com`) hoàn toàn miễn phí và không cần đăng ký tài khoản.
- **Hỗ trợ Vision & Ngữ cảnh 128K**: Tích hợp sẵn Flash Attention (`-fa on`), nén KV cache 8-bit (`q8_0`), tùy chọn MTP (`--spec-type draft-mtp`), nhận ảnh (`mmproj-Q8_0`) và bật/tắt chế độ suy luận (Thinking).
- **Tự động giữ kết nối & tự khôi phục**: Kết hợp JavaScript chống ngủ trình duyệt, nhịp heartbeat ra stdout, xung GPU 1-token có kèm API Key (tránh bị cắt phiên do GPU nhàn rỗi) và tự khởi động lại nếu `llama-server` hoặc tunnel bị ngắt.

---

## So sánh Ngrok và Cloudflare Quick Tunnel

Bạn có thể chuyển đổi linh hoạt qua biến `TUNNEL_PROVIDER` ở Cell 1 của Notebook 2:

| Tiêu chí | **Ngrok** (`"ngrok"`) | **Cloudflare Quick Tunnel** (`"cloudflare"`) |
| :--- | :--- | :--- |
| **Cần tài khoản / Secrets?** | Có (`NGROK_AUTHTOKEN`) | **Không** (chạy ngay lập tức, không cần cấu hình) |
| **Kiểu tên miền công khai** | Hỗ trợ **tên miền tĩnh cố định** (`NGROK_DOMAIN`) | **Tên miền ngẫu nhiên tạm thời** (`*.trycloudflare.com`), đổi mới mỗi lần khởi động lại |
| **Phù hợp khi nào?** | Muốn lưu 1 Base URL cố định trên Cursor, Cline, Open WebUI hoặc app điện thoại | Muốn chạy nhanh không cần tạo tài khoản Ngrok hay thêm Secrets |
| **Chế độ mặc định (`"auto"`)** | Tự chọn nếu phát hiện `NGROK_AUTHTOKEN` | Tự động dùng làm phương án dự phòng nếu thiếu `NGROK_AUTHTOKEN` |

---

## Hướng dẫn chạy nhanh từng bước

### Bước 1 — Tải sẵn mô hình trên CPU (Notebook 1)

#### Trên Kaggle
1. Tải `notebook_1_cpu_prep_keepalive_en.ipynb` lên Kaggle.
2. Ở mục **Session options**, chọn **Accelerator: None (CPU)** và bật **Internet: ON**.
3. Nhấn **Save Version** → **Save & Run All**.
4. Sau khi chạy xong, vào tab **Output** và lưu thư mục `/kaggle/working/bonsai_bundle` thành **Private Dataset**.

#### Trên Google Colab
1. Tải `notebook_1_cpu_prep_keepalive_en.ipynb` lên Google Colab.
2. Chọn **Runtime** → **Change runtime type** → **CPU**.
3. Chạy cell và cấp quyền truy cập Google Drive. Toàn bộ tệp sẽ lưu tại `/content/drive/MyDrive/bonsai_bundle`.

### Bước 2 — Khởi chạy máy chủ GPU (Notebook 2)

#### Trên Kaggle
1. Tải `notebook_2_gpu_server_keepalive_en.ipynb` lên Kaggle.
2. Gắn Dataset vừa tạo ở Bước 1 qua **Add Input** → **Your Datasets**.
3. Ở **Session options**, chọn **Accelerator: GPU T4 x2** và bật **Internet: ON**.
4. Vào **Add-ons** → **Secrets** và thiết lập:
   - `API_KEY` *(khuyên dùng)*: Khóa bí mật của riêng bạn (nếu bỏ trống sẽ tự sinh ngẫu nhiên).
   - `NGROK_AUTHTOKEN` *(tùy chọn)*: Chỉ cần nếu muốn dùng Ngrok; bỏ qua để tự dùng Cloudflare Quick Tunnel miễn phí.
   - `NGROK_DOMAIN` *(tùy chọn)*: Tên miền tĩnh Ngrok của bạn.
5. Nhấn **Run All**.

#### Trên Google Colab
1. Tải `notebook_2_gpu_server_keepalive_en.ipynb` lên Google Colab.
2. Chọn **Runtime** → **Change runtime type** → **T4 / L4 / A100**. *(Nếu dùng 1 card T4 16 GB và bị lỗi tràn VRAM, hãy hạ `CONTEXT_SIZE` xuống `32768` ở Cell 1.)*
3. Thêm `API_KEY` (và Secrets Ngrok nếu dùng) tại biểu tượng **Chìa khóa (Secrets)** bên trái rồi bật **Notebook access**.
4. Đặt `COLAB_USE_GOOGLE_DRIVE = True` ở Cell 1 nếu bạn đã chạy Notebook 1 trên Colab.
5. Chạy lần lượt Cell **1 → 3**.

---

## Kết nối ứng dụng AI (Client)

Sao chép thông tin in ra ở cuối Cell 3 của Notebook 2 để dán vào các ứng dụng hỗ trợ chuẩn OpenAI (Open WebUI, Cursor, Cline, Continue, RikkaHub...):

- **Base URL**: `https://<ten-mien-tunnel-cua-ban>/v1`
- **API Key**: `API_KEY` bạn đã đặt (hoặc khóa `sk-bonsai-...` hệ thống tự sinh)
- **Model ID**: `Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf`

### Ví dụ gọi bằng Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-TUNNEL-URL/v1",
    api_key="YOUR_API_KEY",
)

response = client.chat.completions.create(
    model="Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf",
    messages=[{"role": "user", "content": "Xin chào! Xác nhận mô hình đang hoạt động."}],
)
print(response.choices[0].message.content)
```

### Kiểm tra nhanh lớp bảo mật bằng `curl`

```bash
# Phải trả về 401 Unauthorized (bị chặn khi không có API Key)
curl -i https://YOUR-TUNNEL-URL/v1/models

# Phải trả về 200 OK (hợp lệ khi có Bearer token)
curl -s https://YOUR-TUNNEL-URL/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Bảng giải thích các biến cấu hình chính (Notebook 2, Cell 1)

| Biến | Mặc định | Công dụng |
| :--- | :--- | :--- |
| `USE_MTP` | `True` | Bật dự đoán đa token (Multi-Token Prediction) giúp sinh chữ nhanh hơn. Đổi thành `False` nếu gửi ảnh bị lỗi. |
| `USE_VISION` | `True` | Nạp `mmproj-Q8_0.gguf` để đọc hình ảnh. Đổi thành `False` nếu chỉ chat chữ để tiết kiệm VRAM. |
| `IMAGE_MIN_TOKENS` | `1024` | Số token ảnh tối thiểu (`--image-min-tokens`), giúp đọc chữ nhỏ/chi tiết trong ảnh rõ hơn. |
| `CONTEXT_SIZE` | `131072` | Độ dài ngữ cảnh tối đa (128K). Hạ xuống `32768` hoặc `65536` nếu chạy trên 1 GPU 16 GB bị OOM. |
| `KV_CACHE_TYPE` | `"q8_0"` | Nén bộ nhớ KV cache 8-bit để chứa ngữ cảnh dài trong VRAM. |
| `DEFAULT_THINKING` | `False` | `False` tắt chế độ suy luận mặc định (`--reasoning-budget 0`); `True` bật suy luận theo mức `REASONING_EFFORT`. |
| `REQUIRE_API_KEY` | `True` | Bắt buộc truyền `--api-key` vào `llama-server` để chặn người lạ dùng ké mô hình dù biết Base URL. |
| `API_KEY` | `""` | Khóa nhập tay tùy chọn (khuyên lưu trong Secrets; tự sinh ngẫu nhiên nếu cả hai đều trống). |
| `SHOW_API_KEY_IN_OUTPUT` | `False` | Ẩn bớt ký tự khóa lấy từ Secrets trong log (khóa tự sinh vẫn luôn hiện 1 lần để bạn sao chép). |
| `TUNNEL_PROVIDER` | `"auto"` | `"auto"` (ưu tiên Ngrok nếu có token, thiếu thì dùng Cloudflare), `"ngrok"` hoặc `"cloudflare"`. |
| `ENABLE_KEEP_ALIVE` | `True` | Bật vòng lặp kiểm tra sức khỏe, xung GPU và tự khởi động lại ở Cell 3. |
| `KEEP_ALIVE_MIN_HOURS` | `5.0` | Số giờ duy trì tối thiểu trước khi vòng lặp tự dừng. |
| `KEEP_ALIVE_GPU_PULSE_SEC` | `180` | Chu kỳ (giây) gửi 1 request 1-token có kèm API Key để tránh bị cắt phiên do GPU nhàn rỗi. |

---

## Xử lý sự cố thường gặp

- **Lỗi tràn VRAM (CUDA OOM) trên 1 card T4 (16 GB)**: Hạ `CONTEXT_SIZE` từ `131072` xuống `32768` (hoặc `65536`), hoặc đặt `USE_VISION = False` nếu không cần nhận ảnh.
- **Gửi ảnh bị lỗi khi bật MTP**: Đổi `USE_MTP = False` ở Cell 1 (và bật `INCLUDE_PLAIN_NO_MTP = True` ở Notebook 1 nếu muốn tải sẵn bản không MTP).
- **Tắt máy chủ sạch sẽ**: Bỏ dấu `#` trước dòng `# stop_all()` ở Cell 4 rồi chạy để đóng gọn `llama-server`, `ngrok` và `cloudflared`.

---

## Nguồn tham khảo

- Trọng số mô hình: `BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP-GGUF` (Hugging Face)
- Engine chạy suy luận: `PrismML-Eng/llama.cpp` (`prism-b10828-a153697`)
