# Image Caption Generation API Documentation

## Overview

The **Image Caption Generation API** allows users to upload an image and generate captions for it in Bengali. It is built using **FastAPI** and utilizes a machine learning model to produce captions.

## Base URL

```text
http://0.0.0.0:5000
```

---

## Endpoints

### 1. Root Endpoint

**Description**: Confirms that the API is live.

**Method**: `GET`

**URL**: `/`

**Response**:

- **Status Code**: `200`
- **Body**:

  ```json
  {
      "message": "Image Caption Generation in Bengali"
  }
  ```

---

### 2. Upload Image

**Description**: Uploads an image for caption generation. This replaces any existing image in the upload folder.

**Method**: `POST`

**URL**: `/upload`

**Request**:

- **File**: `image`
  - **Accepted Formats**: `jpg`, `jpeg`, `png`

**Response**:

- **Status Code**: `200`
- **Body**:

  ```json
  {
      "message": "Image uploaded successfully.",
      "filename": "uploaded_images/image.<extension>"
  }
  ```

**Errors**:

- **Status Code**: `400`
  - If no file is selected:

    ```json
    {
        "detail": "No file selected."
    }
    ```

  - If the file format is invalid:

    ```json
    {
        "detail": "Invalid file format. Only jpg, jpeg, png are supported."
    }
    ```

---

### 3. Generate Caption

**Description**: Generate a caption for an image. Two workflows are supported:

- Single-step (recommended): POST the image directly to `/caption` and receive the caption plus token-level and attention outputs in the response.
- Two-step (legacy): POST an image to `/upload` and then GET `/caption` to generate a caption for the previously uploaded file (this flow is still supported for backward compatibility).

#### **Single-step (recommended)**

**Method**: `POST`

**URL**: `/caption`

**Request**:

- Content-Type: multipart/form-data
- File field name: `image` (accepted formats: `jpg`, `jpeg`, `png`)

**Successful Response**:

- **Status Code**: `200`
- **Body**: JSON object with the generated caption and optional token-level outputs and attention image.

Example shape:

  ```json
  {
    "filename": "image.jpg",
    "caption": "একটি সুন্দর ছবি",
    "token_ids": [20021, 16452, 15838],
    "tokens": ["একটি", "সুন্দর", "ছবি"],
    "token_scores": [0.0000478, 0.0000478, 0.0000478],
    "attention_image": "<base64-encoded-png>",
    "attention_image_bytes": "<base64-encoded-png>",
    "attention_means": [0.6692399, 0.47280413, 0.4765392],
    "attention_topk": [
      [[6,2,0.999992],[7,6,0.985347],...],
      [[0,6,0.999996],[0,7,0.842281],...],
      ...
    ],
    "attention_topk_items": [
      [{"row":6,"col":2,"score":0.999992},{"row":7,"col":6,"score":0.985347},...],
      [{"row":0,"col":6,"score":0.999996},{"row":0,"col":7,"score":0.842281},...],
      ...
    ],
    "attention_grid": [8, 8],
    "attention_shape": {"rows": 8, "cols": 8}
  }
  ```

- `attention_image` is a base64-encoded PNG (single composite overlay) if attention visualization was produced; it can be `null` if plotting is unavailable in the runtime.
- `attention_image_bytes` is provided as the same base64-encoded PNG string for clients that expect an `_bytes` key (decode on client to obtain raw PNG bytes).
- `attention_means` is a list of per-token mean attention scores (floats) over the spatial grid.
- `attention_topk` is a nested list providing, for each token, the top-k spatial locations with their scores as `[row, col, score]` entries.
- `attention_topk_items` is a nested list of objects for each token; each object has keys `{ "row": <int>, "col": <int>, "score": <float> }`. This format is convenient for clients (e.g., Flutter) that map top-k items to marker positions.
- `attention_grid` is an optional two-element list `[rows, cols]` giving the spatial grid dimensions the attention maps were computed on (useful for overlay markers).
- `attention_shape` is an optional map `{ "rows": <int>, "cols": <int> }` duplicating the same information for clients that prefer named fields.

## Client decoding notes (Flutter)

- The server returns `attention_image_bytes` as a base64-encoded string (JSON-safe). On the Flutter side decode it like:

```dart
import 'dart:convert';
import 'dart:typed_data';

String b64 = response['attention_image_bytes'] as String;
Uint8List bytes = base64Decode(b64);
// Use Image.memory(bytes) in the widget tree to display the overlay image.
```

- `attention_grid` and `attention_shape` are derived from the attention map length. If the attention keys aren't square, the server will set `attention_grid` to `[1, key_len]` and `attention_shape` to `{ "rows": 1, "cols": key_len }`. Clients should handle this case when computing cell widths/heights for overlay markers.

- Mapping `attention_topk_items` to `TopKItem` instances (example):

```dart
// Assuming TopKItem has a constructor TopKItem({required int row, required int col, required double score});
final rawTopk = response['attention_topk_items'];
List<List<TopKItem>>? topk;
if (rawTopk is List) {
  try {
    topk = rawTopk
        .map<List<TopKItem>>((outer) => (outer as List)
            .map<TopKItem>((it) => TopKItem(row: it['row'] as int, col: it['col'] as int, score: (it['score'] as num).toDouble()))
            .toList())
        .toList();
  } catch (_) {
    topk = null;
  }
}
```

**Errors**:

- **Status Code**: `400` — If no file is provided or file format is invalid. Example:

  ```json
  { "detail": "No file selected." }
  ```

- **Status Code**: `500` — If caption generation fails. Example:

  ```json
  { "detail": "Caption generation failed: <error_message>" }
  ```

**Notes and client guidance**:

- The single-step `POST /caption` is the recommended flow for mobile clients (for example, Flutter apps). It avoids needing a separate upload step.
- Clients should handle the possibility that `attention_image` is missing or `null` (server-side plotting may fall back to a non-image output depending on environment).
- Token-level arrays (`token_ids`, `tokens`, `token_scores`) are included to enable client-side display of token confidences and to support stepwise UI features.

---

## Configuration Details

### Logging

The API uses **loguru** for logging. Logs are stored in the `logs/app.log` file with the following configuration:

- **Rotation**: Daily
- **Retention**: 7 days
- **Compression**: `zip`

### CORS

CORS middleware is configured to allow requests from any origin. This is useful for development and testing.

---

## Upload Folder

The uploaded images are stored in the `uploaded_images` directory. This folder is cleared each time a new image is uploaded.

---

## Running the API

The API can be run using the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

---

## Error Handling

The API handles errors gracefully with appropriate HTTP status codes and detailed error messages. Logs for errors are stored in the `logs/app.log` file for debugging purposes.

---

## Example Workflow

1. **Check API Status**:
   - Send a `GET` request to `/`.
   - Response:

     ```json
     {
         "message": "Image Caption Generation in Bengali"
     }
     ```

2. **Upload an Image**:
   - Send a `POST` request to `/upload` with an image file.
   - Response:

     ```json
     {
         "message": "Image uploaded successfully.",
         "filename": "uploaded_images/image.<extension>"
     }
     ```

3. **Generate Caption**:
   - Send a `GET` request to `/caption`.
   - Response:

     ```json
     {
         "image": "image.<extension>",
         "caption": "<generated_caption>"
     }
     ```

---

## Notes

- Ensure the ML model and vocabulary files are correctly loaded in the `ImgCap/captioner.py` module.
- Logs are crucial for identifying and debugging issues.

---
