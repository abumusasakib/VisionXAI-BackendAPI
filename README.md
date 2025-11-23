# **Image Captioning API**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This project provides an API to generate captions for images using a pre-trained image captioning model. The application is built with **FastAPI** and supports deployment via **Docker**, including **pyenv** support for Python version management and weight transfer support via PowerShell.

---

## Example cURL (single-step - recommended)

  ```bash
  # POST the image and get enriched JSON (caption, tokens, attention, ...)
  curl -X POST "http://127.0.0.1:5000/caption" -F "image=@path_to_image.jpg" -o response.json
  jq . response.json    # pretty-print (optional)
  ```

### Two-step flow (upload then GET)

  curl -X POST "<http://127.0.0.1:5000/upload>" -F "image=@path_to_image.jpg"
  curl -X GET "<http://127.0.0.1:5000/caption>"

### Expected JSON response shape from POST `/caption` (single-step)

The API returns an enriched JSON object. Minimal example (happy path):

  ```json
  {
    "filename": "image.jpg",
    "caption": "একটি সুন্দর ছবি",
    "token_ids": [20021,16452,15838],
    "tokens": ["একটি","সুন্দর","ছবি"],
    "token_scores": [0.0000478,0.0000478,0.0000478],
    "attention_image": "<base64-png-or-null>",
    "attention_means": [0.6692399,0.47280413,0.4765392],
    "attention_topk": [
      [[6,2,0.999992],[7,6,0.985347],...],
      [[0,6,0.999996],[0,7,0.842281],...],
      ...
    ]
  }
  ```

  Notes:
    - `attention_image` may be a base64-encoded PNG or `null` depending on server-side plotting availability. Clients should handle both cases.
    - `token_ids` / `tokens` / `token_scores` allow building token-level UIs (confidence bars, per-token animations, etc.).
    - `attention_colors` (optional): the server may also return `attention_colors`, a list of hex color strings aligned with the `tokens` list (for example `["#1f77b4", "#aec7e8", ...]`).
      Clients can use this list or the convenience map `attention_color_map` (token -> hex) to deterministically color per-token markers when rendering attention overlays. These fields are `null` if the server could not compute colors in the runtime environment.

## **Folder Structure**

```text
.
├── .gitignore
├── .python-version          # Managed by pyenv/pyenv-win
├── .vscode/
│   └── launch.json
├── API_DOCUMENTATION.md
├── Dockerfile
├── ImgCap/
│   ├── __init__.py
│   ├── captioner.py
│   └── weights/
│       ├── checkpoint
│       ├── imgcap_231005.data-00000-of-00001
│       ├── imgcap_231005.index
│       ├── readme.txt
│       └── vocab_231005
├── README.md
├── docker-compose.yml
├── install-pyenv-win.ps1
├── logs/
│   └── *.log.zip
├── main.py
├── managed_context/
│   └── metadata.json
├── requirements.txt
├── setup.bat / setup.sh / setup.ps1
├── test_suite_analysis/
│   └── metadata.json
├── transfer_weight_files.ps1       # PowerShell script for remote model file sync
└── .env.example              # Template for the .env file
```

---

## ⚙️ Setup Instructions

### ▶️ Using Python Locally

Recommended Python version: **3.8.5**

1. **Create a Virtual Environment**:

   ```bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment**:

   * On macOS/Linux:

     ```bash
     source .venv/bin/activate
     ```

   * On Windows:

     ```cmd
     .venv\Scripts\activate
     ```

3. **Run the Application**:

   ```bash
   python main.py
    // server returns enriched JSON (caption, tokens, attention_image, ...)

4. **Dart outline to parse backend output**:

    ```python
    final caption = map['caption'] as String?;
    final tokens = (map['tokens'] as List?)?.cast<String>();
    final tokenScores = (map['token_scores'] as List?)?.cast<double>();
    final attentionBase64 = map['attention_image'] as String?;

    // If attention image is present you can decode it client-side.
    return caption ?? '';
    ```

### 🐳 Using Docker

#### On Linux/macO

```bash
chmod +x setup.sh
./setup.sh
```

#### On Windows

* PowerShell:

  ```powershell
  .\setup.ps1
  ```

* Command Prompt:

  ```cmd
  setup.bat
  ```

---

## 🖋️ Fonts installed in the Docker image

The Docker image installs several Bengali fonts during the build so matplotlib can render Bengali script correctly in attention visualizations. During the image build a short verification step runs `fc-list | grep -i bengali` and prints the found fonts into the build log.

### Installed font sources (downloaded during build)

* Noto Sans Bengali (NotoSansBengali)
* Noto Serif Bengali (NotoSerifBengali)
* SolaimanLipi
* Kalpurush
* Mukti

If you want to change which fonts are installed or add additional fonts, edit the `Dockerfile` font-download block near the top of the file. The build steps use `wget`/`unzip` and copy any `.ttf`/`.otf` files into `/usr/share/fonts/truetype/bengali` and run `fc-cache -fv`.

To test which Bengali fonts are present after building the image you can inspect the build output (the Docker build prints the `fc-list | grep -i bengali` results), or run the following inside a running container:

```powershell
docker run --rm -it -v "$(pwd):/app" visionxai-runner:latest bash -lc "fc-list | grep -i bengali || echo 'no bengali fonts found'"
```

If matplotlib still warns about missing glyphs, consider adding a font with broader Bengali coverage (for example an updated Noto Sans/Serif Bengali release) and re-building the image.

---

## Bundled fonts and matplotlib requirement

This repository bundles a Bengali font used for attention visualizations to ensure captions and token labels render correctly even on systems without system-wide Bengali fonts. The bundled font is placed at `ImgCap/fonts/NotoSansBengali-Regular.ttf` and is automatically registered by the server code when available.

* matplotlib is required to generate the attention visualization image. The project pins matplotlib in `requirements.txt` (e.g., `matplotlib==3.7.1`). If matplotlib is not installed, the captioner will fall back to a simpler PIL-based overlay.

If you want system-wide font installation instead (so other apps on the machine can also use Bengali fonts), copy the TTF files from `ImgCap/fonts/` into `C:\Windows\Fonts` on Windows (requires admin privileges) or the appropriate fonts directory on Linux and run `fc-cache -fv`.

---

## 🐍 Python Version Management with pyenv

You can use `pyenv` (Linux/macOS) or `pyenv-win` (Windows) to lock Python to version 3.8.5.

### Linux/macOS

```bash
curl -fsSL https://pyenv.run | bash
```

Or manually:

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
```

### Windows (pyenv-win)

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

Then verify and install:

```powershell
pyenv --version
pyenv install 3.8.5
pyenv global 3.8.5
```

---

## 🔁 Sync Weights from Local Machine

Use `transfer_weight_files.ps1` for syncing model weights via `scp` over SSH:

> Copy `.env.example` to `.env` and fill in your actual values.

Example `.env.example`:

```dotenv
LOCAL_PATH=D:/your/local/path/to/weights
REMOTE_USER=root
REMOTE_HOST=192.168.0.101
REMOTE_PATH=/mnt/dietpi_userdata/visionxai/
```

Then run:

```powershell
.\transfer_weight_files.ps1
```

---

## 🧪 Testing the API

* Swagger UI:
  `http://localhost:5000/docs`

* Example cURL:

  ```bash
  curl -X POST "http://127.0.0.1:5000/upload" -F "image=@path_to_image.jpg"
  curl -X GET "http://127.0.0.1:5000/caption"
  ```

---

## 🐞 Debugging with VS Code

`.vscode/launch.json` contains configurations for:

* Run with FastAPI
* Debug with FastAPI

Both set `PYTHONUNBUFFERED=1` for clean logs.

---

## 📚 Resources

* [FastAPI Docs](https://fastapi.tiangolo.com/)
* [Uvicorn Docs](https://www.uvicorn.org/)
* [Pyenv](https://github.com/pyenv/pyenv)
* [Pyenv-Win](https://github.com/pyenv-win/pyenv-win)

---

> **Note**: This project is for educational/demo purposes. Production deployments may require additional security and performance hardening.

---

## 📱 Flutter client example (simple usage)

Below is a minimal example showing how a Flutter app can call the API's POST `/caption` endpoint to upload an image and receive the generated caption.

### Add `http` to your `pubspec.yaml`

```yaml
dependencies:
  http: ^0.13.6
```

### Example Dart code to upload an image (e.g., from `image_picker`) and get a caption

```dart
import 'dart:io';
import 'package:http/http.dart' as http;

Future<String> uploadAndCaption(File imageFile) async {
  final uri = Uri.parse('http://YOUR_SERVER_IP:5000/caption');
  final request = http.MultipartRequest('POST', uri);
  request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

  final streamed = await request.send();
  final resp = await http.Response.fromStream(streamed);

  if (resp.statusCode == 200) {
    // server returns JSON: { "filename": "...", "caption": "..." }
    final map = jsonDecode(resp.body);
    return map['caption'] as String;
  } else {
    throw Exception('Caption failed: ${resp.statusCode} ${resp.body}');
  }
}
```

### Expected JSON response shape from POST `/caption`

```json
{
  "filename": "image.jpg",
  "caption": "একটি সুন্দর ছবি ..."
}
```

If you prefer the two-step flow (upload then GET /caption), call POST `/upload` first, then call GET `/caption` to fetch the caption for the previously uploaded file.

---

## 🧰 Tools

This repository includes several small helper scripts in the `tools/` directory to assist with local testing and verification:

* `tools/run_inference_sample.py`: Runs a local, in-Python inference invocation by importing `ImgCap.captioner.generate_from_bytes`. It looks for an image in `uploaded_images/`, runs the captioner, and writes results to `tools/response.json`. If an attention image is produced it is written to `tools/attention_out.png`.

  Usage (from repository root):

  ```powershell
  python tools/run_inference_sample.py
  ```

  Notes:
  * This script imports your local Python environment's packages; ensure dependencies from `requirements.txt` are installed.
  * The script will print a JSON summary to stdout and save `tools/response.json` on success.

* `tools/check_tensorflow_weights.py`: A small utility to inspect TensorFlow checkpoint contents and report shapes/keys. Use it to verify that the model weight files under `ImgCap/weights/` are readable by TensorFlow.

If you run into errors when running these tools (missing packages, incompatible checkpoint formats), see the README's setup instructions and ensure you have the required packages installed and the correct model files present under `ImgCap/weights/`.
