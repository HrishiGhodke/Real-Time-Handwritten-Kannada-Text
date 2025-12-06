# Real-Time Handwritten Kannada Text Recognizer

This repository implements an end-to-end system for **offline handwritten Kannada character recognition** using the **Chars74K** dataset and a **ResNet-18** model. The current version recognizes **single characters** and returns the corresponding **Kannada character output**. Future work will extend this to word- and sentence-level recognition.

## 1. Features

- Uses the **Chars74K Kannada Handwritten subset** for training.
- Transfer learning with **ResNet-18** (PyTorch).
- Mixed precision training and early stopping.
- Completed **Unicode mapping** from model output indices to Kannada characters via `class_mapping.json`.
- Inference notebook for quick testing on sample images.

## 2. Repository Structure

- `Training.ipynb`  
  Training notebook:
  - Extracts Chars74K Kannada archives.
  - Reads `Lists/Kannada/Hnd/all.txt` to load image paths and labels.
  - Builds train/test splits.
  - Defines `KannadaDataset` and `DataLoader`s.
  - Sets up and trains ResNet-18 for 657 Kannada classes.
  - Saves best model weights to `best_model.pth`.

- `run_project.ipynb`  
  Inference / demo notebook:
  - Loads the trained model from `best_model.pth`.
  - Loads class-to-Unicode mapping from `class_mapping.json`.
  - Provides a `predict_image(img_path)` function for testing.
  - Displays the image with the predicted Unicode Kannada character.

- `class_mapping.json`  
  - JSON dictionary mapping **class indices** → **Kannada Unicode labels**.
  - Generated once from the official Chars74K mapping resources (`map.mat` / `source_image_map.txt`).
  - Used at inference time to convert numeric model outputs into proper Kannada text.

- `chars74k/`  
  Extracted Chars74K dataset directory (not committed if too large). Contains:
  - `Kannada/Hnd/Img/` – handwritten character images.
  - `Lists/`, `ListsTXT/` – index lists for splits.

- `best_model.pth`  
  Trained ResNet-18 checkpoint.

## 3. Installation

> Recommended: Python 3.9+ and a GPU (optional, but helpful for training).

If you are using Google Colab, most core packages (PyTorch, torchvision, NumPy, Matplotlib) are already available. You may only need to install a few extras (e.g., `tqdm`, `scikit-learn`).

## 4. Training the Model (`Training.ipynb`)

1. Download and extract the Chars74K dataset (Kannada subsets) from the official page:
  - https://teodecampos.github.io/chars74k/
  - Place the extracted Kannada folders into a directory named `chars74k/` in the project root, or adjust paths in the notebook accordingly.
2. Open `Training.ipynb` (in VS Code, Jupyter, or Colab).
3. Run the cells in order:
   - Dataset extraction and listing.
   - Dataset loading and `DataLoader` creation.
   - Model definition (ResNet-18) and optimizer setup.
   - Training loop with mixed precision and early stopping.
4. When training completes, the best model is saved as `best_model.pth` in the project root.

## 5. Running Inference (`run_project.ipynb`)

1. Make sure the following exist in the project directory:
   - `best_model.pth` – trained model weights.
   - `class_mapping.json` – mapping from class indices to Unicode Kannada characters.
   - `source_image_map.txt` - checks mapping, and if not, creates it
   - Extracted dataset under `chars74k/` (for sample images) and `Maps/` for mapping generation.

2. Open `run_project.ipynb`.

3. Run the cells in order:
   - **Step 1:** Imports and device setup.
   - **Step 2:** Generate `class_mapping.json` from `Maps/Kannada/Img/source_image_map.txt` if it does not already exist, then load it into `id2char`.
   - **Step 3:** Load the trained ResNet-18 model from `best_model.pth` and switch to evaluation mode.
   - **Step 4:** Call `predict_image(img_path)` on a sample image, for example:

     ```python
     image_path = r"chars74k\Kannada\Hnd\Img\Sample001\img001-001.png"
     predict_image(image_path)
     ```

   - The notebook will show the input image and print the predicted Kannada Unicode character.

## 6. Unicode Mapping (Completed)

During inference, the model outputs an integer class index `c`. The file `class_mapping.json` stores a mapping such as:

```json
{
  "0": "ಅ",
  "1": "ಆ",
  "2": "ಇ",
  ...
}
```

The inference code:

```python
with open("class_mapping.json", "r", encoding="utf-8") as f:
    id2char = {int(k): v for k, v in json.load(f).items()}

predicted_class = outputs.argmax(dim=1).item()
predicted_char = id2char.get(predicted_class, f"Class_{predicted_class+1}")
```

This provides a **complete Unicode mapping** from model outputs to actual Kannada script, enabling true OCR output.

## 7. Future Work: Words and Sentences

Currently, the system recognizes **one character at a time**. Planned enhancements:

1. **Word-level recognition:**
   - Segment an input word image into characters.
   - Run each character through this model.
   - Combine predicted characters to reconstruct the word.

2. **Sentence-level recognition:**
   - Segment a sentence image into words and characters.
   - Apply character recognition to each segment.
   - Reconstruct full sentences in Unicode Kannada.

3. **Sequence models (future extension):**
   - Replace explicit segmentation with CNN + BiLSTM / Transformer + CTC-based models to recognize complete words/lines directly.

## 8. Acknowledgments

- Chars74K dataset authors and maintainers.
- PyTorch and broader open-source community.
- Academic guidance and institutional support involved in this project.
