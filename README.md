<p align="center">
  <strong>🇺🇸 English</strong> | <a href="./README.vi.md">🇻🇳 Tiếng Việt</a>
</p>

# Ternary-Bonsai-2-27B Cloud Inference Server (Kaggle & Google Colab)

Run **Ternary-Bonsai-2-27B** (`PQ2_0-MTP` + Vision) as a private, OpenAI-compatible API server on free cloud GPUs (**Kaggle T4 x2** or **Google Colab T4 / L4 / A100**) using PrismML's CUDA 12.8 `llama-server`.

This repository splits the workflow into **two notebooks** so you spend **0 GPU hours** downloading multi-gigabyte weights and launch the GPU server in **~30 seconds** with built-in **API Key protection**, **Ngrok + Cloudflare Quick Tunnel** support, and **Auto Keep-Alive**.

---

## Repository Contents

| File | Accelerator | Purpose |
| :--- | :--- | :--- |
| `notebook_1_cpu_prep_keepalive_en.ipynb` | **CPU Only (0 GPU hrs)** | Pre-downloads the PrismML CUDA 12.8 `llama-server` binary, the `Ternary-Bonsai-2-27B` GGUF model (7.66 GB), and the Vision projector (`mmproj-Q8_0.gguf`, 0.63 GB) into a reusable Kaggle Dataset or Google Drive folder. |
| `notebook_2_gpu_server_keepalive_en.ipynb` | **GPU (T4 x2 / T4 / L4 / A100)** | Loads the pre-downloaded bundle in ~30s, starts `llama-server` with `--api-key` authentication, exposes it via **Ngrok** or **Cloudflare Quick Tunnel**, and maintains uptime with an automated Keep-Alive loop. |

---

## Key Features

- **Zero GPU Quota Wasted on Downloads**: Download once on CPU in Notebook 1 and reuse instantly across sessions.
- **Built-in API Key Authentication (`--api-key`)**: Protects `/v1/chat/completions`, `/v1/models`, and all model routes with Bearer token authentication (`HTTP 401` for unauthorized requests). If you leave `API_KEY` empty while `REQUIRE_API_KEY = True`, a strong random `sk-bonsai-...` key is auto-generated at startup and verified automatically.
- **Dual Tunnel Support (Ngrok + Cloudflare Fallback)**: Set `TUNNEL_PROVIDER = "auto"` (default) to use **Ngrok** when `NGROK_AUTHTOKEN` is configured, or automatically fall back to a free, zero-signup **Cloudflare Quick Tunnel** (`https://<random>.trycloudflare.com`).
- **Vision & 128K Context Ready**: Pre-configured with Flash Attention (`-fa on`), 8-bit KV cache (`q8_0`), optional MTP speculative drafting (`--spec-type draft-mtp`), Vision support (`mmproj-Q8_0`), and toggleable thinking mode.
- **Auto Keep-Alive & Self-Healing**: Combines browser anti-idle JavaScript, rolling stdout heartbeats, authenticated 1-token GPU pulses (to prevent GPU-idle termination), and automatic restarts if `llama-server` or the tunnel drops.

---

## Ngrok vs. Cloudflare Quick Tunnel

Choose your preferred tunnel via `TUNNEL_PROVIDER` in Cell 1 of Notebook 2:

| Feature | **Ngrok** (`"ngrok"`) | **Cloudflare Quick Tunnel** (`"cloudflare"`) |
| :--- | :--- | :--- |
| **Account / Secrets Required** | Yes (`NGROK_AUTHTOKEN`) | **No** (zero setup, works immediately) |
| **URL Stability** | Supports a **fixed static domain** (`NGROK_DOMAIN`) | **Random temporary domain** (`*.trycloudflare.com`) on each restart |
| **Best For** | Saving one permanent Base URL in Cursor, Cline, Open WebUI, or mobile apps | Fast temporary usage without configuring Ngrok Secrets |
| **Default (`"auto"`)** | Used automatically if `NGROK_AUTHTOKEN` is set | Used automatically as a fallback if `NGROK_AUTHTOKEN` is omitted |

---

## Quick Start Guide

### Step 1 — Pre-Download Assets on CPU (Notebook 1)

#### On Kaggle
1. Upload `notebook_1_cpu_prep_keepalive_en.ipynb` to Kaggle.
2. In **Session options**, set **Accelerator** to **None (CPU)** and **Internet** to **ON**.
3. Click **Save Version** → **Save & Run All**.
4. Once finished, open the **Output** tab and save `/kaggle/working/bonsai_bundle` as a **Private Dataset**.

#### On Google Colab
1. Upload `notebook_1_cpu_prep_keepalive_en.ipynb` to Google Colab.
2. Set **Runtime** → **Change runtime type** → **CPU**.
3. Run the cell and authorize Google Drive access. Files are saved to `/content/drive/MyDrive/bonsai_bundle`.

### Step 2 — Launch the GPU Server (Notebook 2)

#### On Kaggle
1. Upload `notebook_2_gpu_server_keepalive_en.ipynb` to Kaggle.
2. Attach your Dataset from Step 1 via **Add Input** → **Your Datasets**.
3. In **Session options**, set **Accelerator** to **GPU T4 x2** and **Internet** to **ON**.
4. Open **Add-ons** → **Secrets** and configure:
   - `API_KEY` *(recommended)*: Your private API key (or leave empty to auto-generate one).
   - `NGROK_AUTHTOKEN` *(optional)*: Only needed for Ngrok; omit to use a free Cloudflare Quick Tunnel automatically.
   - `NGROK_DOMAIN` *(optional)*: Your static Ngrok domain.
5. Click **Run All**.

#### On Google Colab
1. Upload `notebook_2_gpu_server_keepalive_en.ipynb` to Google Colab.
2. Set **Runtime** → **Change runtime type** → **T4 / L4 / A100**. *(On a single 16 GB T4, lower `CONTEXT_SIZE` to `32768` in Cell 1 if you hit out-of-memory errors.)*
3. Add `API_KEY` (and optional Ngrok Secrets) via the **Key icon (Secrets)** in the left sidebar and toggle **Notebook access** ON.
4. Set `COLAB_USE_GOOGLE_DRIVE = True` in Cell 1 if you ran Notebook 1 on Colab.
5. Run Cells **1 → 3**.

---

## Connecting Your AI Client

Copy the printed values from Cell 3 of Notebook 2 into any OpenAI-compatible client (Open WebUI, Cursor, Cline, Continue, RikkaHub, etc.):

- **Base URL**: `https://<your-tunnel-url>/v1`
- **API Key**: Your configured `API_KEY` (or the auto-generated `sk-bonsai-...` key)
- **Model ID**: `Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf`

### Python OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-TUNNEL-URL/v1",
    api_key="YOUR_API_KEY",
)

response = client.chat.completions.create(
    model="Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf",
    messages=[{"role": "user", "content": "Hello! Confirm you are running."}],
)
print(response.choices[0].message.content)
```

### Quick `curl` Security Check

```bash
# Should return 401 Unauthorized (blocked without API key)
curl -i https://YOUR-TUNNEL-URL/v1/models

# Should return 200 OK (authorized)
curl -s https://YOUR-TUNNEL-URL/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Key Configuration Variables (Notebook 2, Cell 1)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `USE_MTP` | `True` | Enables Multi-Token Prediction speculative drafting for faster generation. Set `False` if image inputs misbehave. |
| `USE_VISION` | `True` | Loads `mmproj-Q8_0.gguf` for image understanding. Set `False` to save VRAM for text-only usage. |
| `IMAGE_MIN_TOKENS` | `1024` | Minimum visual tokens per image (`--image-min-tokens`) for clearer small-text OCR. |
| `CONTEXT_SIZE` | `131072` | Context window size (128K). Lower to `32768` or `65536` on single 16 GB GPUs if OOM occurs. |
| `KV_CACHE_TYPE` | `"q8_0"` | 8-bit quantized KV cache to fit long contexts into GPU VRAM. |
| `DEFAULT_THINKING` | `False` | `False` disables chain-of-thought reasoning by default (`--reasoning-budget 0`); `True` enables it using `REASONING_EFFORT`. |
| `REQUIRE_API_KEY` | `True` | Enforces `--api-key` on `llama-server` so public URLs cannot be used without your key. |
| `API_KEY` | `""` | Optional manual key (prefer storing `API_KEY` in Secrets; auto-generated if both are empty). |
| `SHOW_API_KEY_IN_OUTPUT` | `False` | Masks keys loaded from Secrets in cell logs (auto-generated keys are always printed once). |
| `TUNNEL_PROVIDER` | `"auto"` | `"auto"` (Ngrok if token exists, else Cloudflare), `"ngrok"`, or `"cloudflare"`. |
| `ENABLE_KEEP_ALIVE` | `True` | Runs the automated health-check, GPU pulse, and auto-restart loop in Cell 3. |
| `KEEP_ALIVE_MIN_HOURS` | `5.0` | Target uptime in hours before the monitoring loop exits. |
| `KEEP_ALIVE_GPU_PULSE_SEC` | `180` | Interval (seconds) for sending a 1-token authenticated GPU pulse to prevent GPU-idle disconnects. |

---

## Troubleshooting

- **CUDA Out of Memory (OOM) on 1x T4 (16 GB)**: Lower `CONTEXT_SIZE` from `131072` to `32768` (or `65536`), or set `USE_VISION = False` if you do not need image inputs.
- **Image requests fail with MTP enabled**: Set `USE_MTP = False` in Cell 1 (and ensure `INCLUDE_PLAIN_NO_MTP = True` in Notebook 1 if pre-downloading).
- **Clean shutdown**: Uncomment `# stop_all()` in Cell 4 and run it to terminate `llama-server`, `ngrok`, and `cloudflared` cleanly.

---

## Credits

- Model weights: `BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP-GGUF` (Hugging Face)
- Inference engine: `PrismML-Eng/llama.cpp` (`prism-b10828-a153697`)

<p align="center">
  <strong>🇺🇸 English</strong> | <a href="./README.vi.md">🇻🇳 Tiếng Việt</a>
</p>
