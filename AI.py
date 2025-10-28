import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
# -----------------------
# DESIGN: Dark Futuristic + Animated Gradient + Glassmorphism (DROP-IN)
# Paste this block immediately after: from PIL import Image
# -----------------------
# use the already-imported `st` safely
__st_design_guard = st

__st_design_css = r"""
<style>
/* ---------- Page & Background ---------- */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 10% 10%, #071124 0%, #0b1b2b 25%, #071224 50%, #020409 100%);
  background-attachment: fixed;
  color: #e6f7fb;
  min-height: 100vh;
  font-family: 'Segoe UI', Roboto, Arial, sans-serif;
}

/* animated gradient overlay */
#design-gradient {
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(120deg, rgba(0,255,255,0.06), rgba(171,0,255,0.04), rgba(0,150,255,0.04));
  mix-blend-mode: overlay;
  animation: slidegrad 12s linear infinite;
  z-index: 0;
}
@keyframes slidegrad {
  0% { transform: translateX(-20%); }
  50% { transform: translateX(20%); }
  100% { transform: translateX(-20%); }
}

/* ---------- Glass cards / block container ---------- */
.block-container, .reportview-container .main {
  position: relative;
  z-index: 1;
}
[data-testid="stMarkdownContainer"] {
  background: rgba(8, 12, 16, 0.55);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
  backdrop-filter: blur(8px) saturate(120%);
  border: 1px solid rgba(255,255,255,0.03);
}

/* ---------- Title ---------- */
h1 {
  color: #7ef0ff;
  text-align: center;
  letter-spacing: 0.8px;
  font-weight: 700;
  margin-bottom: 6px;
  text-shadow: 0 6px 18px rgba(46, 255, 255, 0.04);
}
h2, h3 {
  color: #bfefff;
}

/* ---------- File uploader (glass + neon dashed) ---------- */
[data-testid="stFileUploader"] {
  border-radius: 14px !important;
  border: 1px dashed rgba(126, 240, 255, 0.35) !important;
  padding: 14px !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stFileUploader"]:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0, 200, 255, 0.08);
}

/* ---------- Buttons: neon gradient ---------- */
div.stButton > button, button[kind="primary"] {
  background: linear-gradient(90deg, rgba(0,200,255,0.18), rgba(171, 0, 255, 0.18));
  color: #eafcff;
  border: 1px solid rgba(126,240,255,0.18);
  padding: 8px 14px;
  border-radius: 10px;
  font-weight: 600;
  box-shadow: 0 6px 22px rgba(0,200,255,0.06);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
div.stButton > button:hover {
  transform: translateY(-4px) scale(1.02);
  box-shadow: 0 18px 45px rgba(0,200,255,0.14);
}

/* ---------- Cards for results & images ---------- */
.stImage, .element-container, .stMetric {
  border-radius: 12px !important;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.03);
}

/* ---------- Grad-CAM section: highlight ---------- */
[data-testid="stImage"] img {
  border-radius: 10px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.6), 0 0 18px rgba(0,200,255,0.04);
}

/* ---------- Info panels / expander ---------- */
div[role="button"].stExpanderHeader {
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.02);
}

/* ---------- Small UI polish ---------- */
p, li, span, label {
  color: #d9f2f7;
  line-height: 1.5;
}
hr {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,200,255,0.12), transparent);
  margin: 16px 0;
}

/* ---------- Footer hide default + custom footer style ---------- */
footer { visibility: hidden; }
.custom-footer {
  text-align: center;
  font-size: 0.9rem;
  color: #99dff0;
  margin-top: 22px;
  opacity: 0.9;
}
.custom-footer a { color: #7ef0ff; text-decoration: none; font-weight: 600; }

/* ---------- Subtle neon heading animation ---------- */
@keyframes neonPulse {
  0% { text-shadow: 0 0 6px rgba(126,240,255,0.06); transform: translateY(0px); }
  50% { text-shadow: 0 0 20px rgba(126,240,255,0.12); transform: translateY(-2px); }
  100% { text-shadow: 0 0 6px rgba(126,240,255,0.06); transform: translateY(0px); }
}
h1 { animation: neonPulse 4.5s ease-in-out infinite; }

/* ---------- scrollbar ---------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #00bcd4, #006b72);
  border-radius: 10px;
}
</style>
"""

# inject gradient overlay element + css
__st_design_html = '<div id="design-gradient"></div>'
__st_design_guard.markdown(__st_design_css + __st_design_html, unsafe_allow_html=True)
# -----------------------
# END OF DESIGN BLOCK
# -----------------------


# -----------------------
# App Configuration
# -----------------------
st.set_page_config(page_title="DermaSense AI", layout="wide")
st.title("DermaSense AI")
st.markdown("""
### Advanced Skin Disease Detection  
Upload a clear image of the affected skin area DermaSense AI will analyze and predict possible conditions using deep learning.

⚠️ **Disclaimer:** This app is for educational and research purposes only. Always consult a dermatologist for medical advice.
""")

# -----------------------
# Load TensorFlow Lite Model
# -----------------------
try:
    tflite_model_path = r"C:\Users\Sahil khan(Machine E\Documents\New folder (4)\model.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    st.success("Loaded TensorFlow Lite model: model.tflite")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    interpreter = None

# -----------------------
# Define Class Labels
# -----------------------
CLASS_NAMES = [
    "Actinic Keratosis (Pre-cancerous rough skin spots from sun exposure)",
    "Basal Cell Carcinoma (Common skin cancer from UV damage)",
    "Dermatofibroma (Benign, firm skin bump, usually harmless)",
    "Nevus (Mole; usually harmless, monitor for changes)",
    "Pigmented Benign Keratosis (Non-cancerous pigmented skin growth)",
    "Seborrheic Keratosis (Benign, waxy or wart-like growths)",
    "Squamous Cell Carcinoma (Aggressive skin cancer from UV exposure)",
    "Vascular Lesion (Skin blood vessel abnormality, e.g., hemangioma)"
]


# -----------------------
# Helper: Skin type heuristics
# -----------------------
def analyze_skin_type(pil_img):
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    h, w = img_rgb.shape[:2]
    img_blur = cv2.GaussianBlur(img_rgb, (5,5), 0)

    img_float = img_blur.astype(np.float32) / 255.0
    r = img_float[..., 0]
    g = img_float[..., 1]
    b = img_float[..., 2]
    red_diff = r - np.maximum(g, b)
    red_diff[red_diff < 0] = 0
    redness_score = np.mean(red_diff)
    redness_prop = np.sum(red_diff > 0.15) / (h*w)

    # Ensure grayscale is float32 to fix Laplacian error
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    bright_mask = gray > 0.85
    bright_prop = np.sum(bright_mask) / (h*w)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture_strength = np.mean(np.abs(lap))
    median_brightness = np.median(gray)

    red_mask = (red_diff > 0.12).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_count = sum(1 for c in contours if cv2.contourArea(c) > 10)

    s_redness = np.clip((redness_score * 3.0) + (redness_prop * 2.0), 0.0, 1.0)
    s_oil = np.clip(bright_prop * 5.0, 0.0, 1.0)
    s_texture = np.clip((texture_strength / 30.0), 0.0, 1.0)
    s_dull = np.clip((0.6 - median_brightness) * 2.5, 0.0, 1.0)
    s_spots = np.clip(spot_count / 30.0, 0.0, 1.0)

    sensitive_score = np.clip(0.6 * s_redness + 0.4 * s_spots, 0.0, 1.0)
    oily_score = np.clip(0.7 * s_oil + 0.3 * (1 - s_texture), 0.0, 1.0)
    dry_score = np.clip(0.6 * s_texture + 0.4 * s_dull, 0.0, 1.0)
    combination_score = np.clip(0.5 * s_oil + 0.5 * s_texture, 0.0, 1.0)
    normal_score = np.clip(1.0 - max(sensitive_score, oily_score, dry_score, combination_score), 0.0, 1.0)

    raw = np.array([normal_score, sensitive_score, oily_score, dry_score, combination_score], dtype=np.float32)
    if raw.sum() == 0:
        probs = np.array([1,0,0,0,0], dtype=np.float32)
    else:
        probs = raw / raw.sum()

    names = ["Normal", "Sensitive", "Oily", "Dry", "Combination"]
    return {names[i]: float(probs[i]) for i in range(len(names))}, {
        "redness_score": float(s_redness),
        "bright_prop": float(bright_prop),
        "texture_strength": float(texture_strength),
        "median_brightness": float(median_brightness),
        "spot_count": int(spot_count)
    }

# -----------------------
# Upload Section
# -----------------------
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)/255.0
    img_array = np.expand_dims(img_array, axis=0)


    # Skin type analysis
    skin_type_probs, debug_metrics = analyze_skin_type(img)
    st.subheader("Skin Type (Heuristic)")
    top_skin = max(skin_type_probs, key=skin_type_probs.get)
    st.markdown(f"**Detected Skin Type:** {top_skin} — {skin_type_probs[top_skin]*100:.1f}% confidence")
    with st.expander("Show detailed skin-type probabilities & metrics"):
        for k, v in skin_type_probs.items():
            st.write(f"- **{k}**: {v*100:.2f}%")
        st.write("**Debug metrics:**")
        st.write(debug_metrics)
        st.info("Note: This is a heuristic estimator. For best accuracy, train a dedicated classifier on labeled skin-type images.")

    # -----------------------
    # Predict with TFLite model
    # -----------------------
    if interpreter is not None:
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            top_indices = predictions.argsort()[-5:][::-1]
            st.subheader("Top Predictions (Condition):")
            for i in top_indices:
                label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
                st.write(f"**{label}** — {predictions[i] * 100:.2f}%")

            top_class = CLASS_NAMES[top_indices[0]] if top_indices[0] < len(CLASS_NAMES) else f"Class {top_indices[0]}"
            st.markdown(f"### 🧬 Most Likely Condition: **{top_class}**") 

            # -----------------------
            # Treatment / Advice
            # -----------------------
            DISEASE_ADVICE = {
                "Actinic Keratosis (Pre-cancerous rough skin spots from sun exposure)": """
1) **Immediate Action:** Protect your skin from further sun exposure—wear long sleeves, hats, and apply SPF 30+ sunscreen daily.  
2) **Consult a Dermatologist:** Early evaluation is important to prevent progression to skin cancer.  
3) **Treatment Options:** Common treatments include cryotherapy (freezing the lesion), topical medicated creams (like 5-fluorouracil or imiquimod), or laser therapy.  
4) **Follow-Up:** Regular skin check-ups to monitor any new or changing lesions.
""",

    "Basal Cell Carcinoma (Common skin cancer from UV damage)": """
1) **Immediate Action:** Avoid excessive sun exposure and protect skin from UV rays.  
2)  **Consult a Dermatologist:** Early diagnosis is critical.  
3) **Treatment Options:** Usually treated surgically (excision or Mohs surgery). Sometimes topical therapy or minor radiation therapy may be recommended.  
4) **Aftercare:** Follow wound care instructions carefully and attend regular follow-ups to monitor recurrence.
""",

    "Dermatofibroma (Benign, firm skin bump, usually harmless)": """
1) **Immediate Action:** Usually no urgent action is needed, as it’s benign.  
2) **Monitor:** Keep track of any changes in size, color, or pain.  
3) **Treatment Options:** If it causes discomfort or cosmetic concern, surgical removal can be performed. No medications are generally required.  
4) **Follow-Up:** Occasional dermatologist check-up if changes occur.
""",

    "Nevus (Mole; usually harmless, monitor for changes)": """
1) **Immediate Action:** Observe your moles for any changes in color, size, shape, or bleeding.  
2) **Consult a Dermatologist:** Especially if you notice changes or new moles appear.  
3) **Treatment Options:** Most are harmless; removal is only necessary if suspicious.  
4) **Skin Care:** Protect moles from sun exposure with sunscreen and clothing.
""",

    "Pigmented Benign Keratosis (Non-cancerous pigmented skin growth)": """
1) **Immediate Action:** Usually no urgent steps needed.  
2) **Consult a Dermatologist:** For reassurance or cosmetic concerns.  
3) **Treatment Options:** Cryotherapy, laser therapy, or curettage can remove lesions if desired.  
4) **Monitoring:** Watch for new or changing spots.
""",

    "Seborrheic Keratosis (Benign, waxy or wart-like growths)": """
1) **Immediate Action:** No immediate concern; generally harmless.  
2) **Monitor:** Observe for irritation, itching, or changes.  
3) **Treatment Options:** If bothersome, cryotherapy, curettage, or laser removal may be used.  
4) **Aftercare:** Minimal recovery time; follow dermatologist instructions if removed.
""",

    "Squamous Cell Carcinoma (Aggressive skin cancer from UV exposure)": """
1) **Immediate Action:** Seek prompt medical attention; this can be serious.  
2) **Consult a Dermatologist/Oncologist:** Early treatment improves outcomes.  
3) **Treatment Options:** Surgical excision, topical chemotherapy, or radiotherapy depending on severity.  
4) **Follow-Up:** Regular skin checks to prevent recurrence and monitor surrounding skin.
""",

    "Vascular Lesion (Skin blood vessel abnormality, e.g., hemangioma)": """
1) **Immediate Action:** Usually harmless, but consult a dermatologist for accurate diagnosis.  
2) **Monitor:** Observe any growth or bleeding.  
3) **Treatment Options:** Laser therapy is the most common; observation is acceptable for minor cases.  
4) **Aftercare:** Follow-up appointments to ensure proper healing and monitor new lesions.
"""
            }

            advice_text = DISEASE_ADVICE.get(top_class, "⚕️ No advice available for this condition. Consult a dermatologist for proper guidance.")
            st.subheader("Suggested Treatment / Advice")
            st.info(advice_text)

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
    else:
        st.warning("Model not loaded skipping condition prediction.")

# -----------------------
# About Me Section
# -----------------------
st.subheader("About Me")
st.markdown("""
I am currently studying **Artificial Intelligence** and have a strong foundation in the field of AI.  
It is my dream to apply AI to **real-world problems** and create something truly **remarkable for humanity** through the power of intelligent technology. 
""")
st.markdown("**Developed by Sahil Khan** | Powered by TensorFlow, OpenCV and Streamlit")
