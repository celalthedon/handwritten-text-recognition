#gradio app.py
#celalibr -> celalthedon
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import gradio as gr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


##model settings
MODEL_OPTIONS = {
    "Base (faster)": "microsoft/trocr-base-handwritten",
    "Large (better quality, heavier)": "microsoft/trocr-large-handwritten",
}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


MODEL_CACHE = {}


def load_trocr(model_name: str):
    if model_name not in MODEL_CACHE:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(get_device())
        model.eval()
        MODEL_CACHE[model_name] = (processor, model)
    return MODEL_CACHE[model_name]


#ocr helper
def segment_lines_from_pil(pil_img: Image.Image, pad: int = 4):
    rgb = pil_img.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    proj = bw.sum(axis=1)
    if proj.max() <= 0:
        return [], bgr, []

    thr = proj.max() * 0.05
    rows = proj > thr

    regions = []
    in_line = False
    start = 0

    for i, r in enumerate(rows):
        if r and not in_line:
            in_line = True
            start = i
        elif not r and in_line:
            in_line = False
            if i - start > 8:
                regions.append((max(0, start - pad), min(bgr.shape[0], i + pad)))

    if in_line and bgr.shape[0] - start > 8:
        regions.append((max(0, start - pad), bgr.shape[0]))

    lines = [rgb.crop((0, y1, rgb.width, y2)) for y1, y2 in regions]
    return lines, bgr, regions


def tight_crop_line(pil_img: Image.Image, margin: int = 3) -> Image.Image:
    arr = np.array(pil_img.convert("L"))
    ink = 255 - arr
    col_sum = ink.sum(axis=0)

    if col_sum.max() <= 0:
        return pil_img

    cols = np.where(col_sum > 0.02 * col_sum.max())[0]
    if len(cols) == 0:
        return pil_img

    x1 = max(0, int(cols[0]) - margin)
    x2 = min(arr.shape[1], int(cols[-1]) + margin + 1)
    return pil_img.crop((x1, 0, x2, arr.shape[0]))


def make_line_variants(pil_img: Image.Image):
    img = tight_crop_line(pil_img).convert("RGB")
    gray = ImageOps.grayscale(img)

    return [
        ("orig", img),
        ("autocontrast", ImageOps.autocontrast(img)),
        ("gray", gray.convert("RGB")),
        ("gray_autocontrast", ImageOps.autocontrast(gray).convert("RGB")),
        ("sharpen", img.filter(ImageFilter.SHARPEN)),
    ]


@torch.no_grad()
def predict_line_trocr_best(
    pil_img: Image.Image,
    processor,
    model,
    max_new_tokens: int = 128
):
    device = get_device()
    candidates = []

    for name, var_img in make_line_variants(pil_img):
        pixel_values = processor(images=var_img, return_tensors="pt").pixel_values.to(device)

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=6,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        score = len([c for c in text if c.isalnum() or c in " .,!?;:'\"-()"])
        candidates.append((score, name, text, var_img))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0], candidates


def annotate_lines(bgr, regions):
    dbg2 = bgr.copy()
    for i, (y1, y2) in enumerate(regions):
        cv2.rectangle(dbg2, (0, y1), (dbg2.shape[1], y2), (0, 200, 0), 2)
        cv2.putText(dbg2, str(i + 1), (8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
    return cv2.cvtColor(dbg2, cv2.COLOR_BGR2RGB)


def recognize_page_trocr(pil_img: Image.Image, processor, model):
    lines, dbg, regions = segment_lines_from_pil(pil_img)
    results = []
    previews = []

    for i, line in enumerate(lines):
        best, candidates = predict_line_trocr_best(line, processor, model)
        _, variant_name, text, _ = best
        results.append(text)
        previews.append({
            "index": i + 1,
            "image": line,
            "variant": variant_name,
            "text": text,
            "candidates": candidates,
        })

    full_text = "\n".join(results)
    annotated = annotate_lines(dbg, regions) if len(regions) else None

    return {
        "lines": lines,
        "regions": regions,
        "annotated": annotated,
        "previews": previews,
        "full_text": full_text,
    }


def build_line_details(previews, show_candidate_variants=False):
    if not previews:
        return "<p>No lines detected.</p>"

    html_parts = []
    for item in previews:
        part = f"""
        <div style="padding:14px; margin:12px 0; border-radius:18px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);">
            <h4 style="margin:0 0 8px 0;">Line {item['index']}</h4>
            <p style="margin:4px 0;"><b>Chosen variant:</b> <code>{item['variant']}</code></p>
            <p style="margin:4px 0;"><b>Prediction:</b> {item['text'] if item['text'] else '[empty]'}</p>
        """

        if show_candidate_variants:
            part += "<div style='margin-top:8px;'><b>Top variants:</b><ul>"
            for score, variant_name, text, _ in item["candidates"][:5]:
                safe_text = text if text else "[empty]"
                part += f"<li><code>{variant_name}</code> | score={score} | {safe_text}</li>"
            part += "</ul></div>"

        part += "</div>"
        html_parts.append(part)

    return "\n".join(html_parts)


def process_image(image, model_label, show_line_cards, show_candidate_variants):
    if image is None:
        return None, "No image uploaded.", "<p>Please upload an image first.</p>"

    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    model_name = MODEL_OPTIONS[model_label]
    processor, model = load_trocr(model_name)
    output = recognize_page_trocr(pil_img, processor, model)

    annotated = output["annotated"]
    full_text = output["full_text"] if output["full_text"].strip() else "No text recognized."

    if show_line_cards:
        details_html = build_line_details(output["previews"], show_candidate_variants)
    else:
        details_html = "<p>Line-by-line results are hidden.</p>"

    return annotated, full_text, details_html



# UI

custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0b1020 0%, #111827 35%, #1f2937 100%) !important;
    color: #f9fafb !important;
    font-family: Arial, sans-serif;
}
#hero {
    padding: 20px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(59,130,246,0.20), rgba(16,185,129,0.18));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-bottom: 18px;
}
.card {
    padding: 16px;
    border-radius: 20px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div id="hero">
            <h1 style="margin-bottom:8px;">✍️ Celalibr Handwriting Reader</h1>
            <p style="margin:0; color:#d1d5db;">
                Creative handwritten OCR app with line segmentation + TrOCR inference.
            </p>
            <p style="margin-top:8px; color:#93c5fd; font-weight:700;">by celalibr</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="card"><h3>Upload handwritten page</h3><p>Best results: clear, high-contrast handwriting with visible line spacing.</p></div>')
            image_input = gr.Image(type="pil", label="Upload an image")

        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="card">
                    <h3>How it works</h3>
                    <ol>
                        <li>Detect text lines from the uploaded page</li>
                        <li>Create several enhanced variants for each line</li>
                        <li>Run TrOCR on each variant</li>
                        <li>Keep the strongest prediction</li>
                        <li>Merge all lines into final text</li>
                    </ol>
                </div>
                """
            )

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(MODEL_OPTIONS.keys()),
            value="Base (faster)",
            label="Model"
        )
        show_line_cards = gr.Checkbox(value=True, label="Show line-by-line results")
        show_candidate_variants = gr.Checkbox(value=False, label="Show top variant guesses")

    read_button = gr.Button("Read handwriting")

    with gr.Row():
        annotated_output = gr.Image(label="Detected lines")
        text_output = gr.Textbox(label="Recognized text", lines=14)

    details_output = gr.HTML(label="Line-by-line results")

    read_button.click(
        fn=process_image,
        inputs=[image_input, model_choice, show_line_cards, show_candidate_variants],
        outputs=[annotated_output, text_output, details_output]
    )

    gr.HTML(
        """
        <div style="text-align:center; padding-top:18px; color:#cbd5e1; font-size:0.95rem;">
            Built with TrOCR, Gradio, OpenCV, and a little creative chaos.<br>
            <strong>by celalibr</strong>
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch()