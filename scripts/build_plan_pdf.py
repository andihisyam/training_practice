from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
QA_DIR = ROOT / "outputs" / "doc_plan_qa"
PDF_PATH = DOCS_DIR / "Struktur_Plan_Penelitian_DM_XAI.pdf"

PAGE_W = 1654
PAGE_H = 2339
MARGIN_X = 110
MARGIN_Y = 110
CONTENT_W = PAGE_W - (2 * MARGIN_X)

BG = "#FFFFFF"
TITLE = "#0F172A"
SUB = "#475569"
BLUE = "#1D4ED8"
BLUE_DARK = "#1E3A8A"
BOX = "#EFF6FF"
BOX_ALT = "#F8FAFC"
LINE = "#CBD5E1"


def load_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(56, bold=True)
FONT_SUBTITLE = load_font(26, bold=False)
FONT_H1 = load_font(30, bold=True)
FONT_H2 = load_font(23, bold=True)
FONT_BODY = load_font(21, bold=False)
FONT_BODY_BOLD = load_font(21, bold=True)
FONT_SMALL = load_font(18, bold=False)


def text_height(font):
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1]


def draw_wrapped(draw, text, xy, font, fill, max_width, line_gap=8):
    x, y = xy
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    line_h = text_height(font)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + line_gap
    return y


def draw_bullet_list(draw, items, x, y, width):
    for item in items:
        bullet_x = x + 4
        text_x = x + 28
        draw.ellipse((bullet_x, y + 9, bullet_x + 8, y + 17), fill=BLUE)
        y = draw_wrapped(draw, item, (text_x, y), FONT_BODY, TITLE, width - 28, line_gap=6)
        y += 8
    return y


def step_box(draw, x, y, w, title, body):
    line1 = text_height(FONT_BODY_BOLD)
    body_top_pad = 20
    inner_pad = 22
    body_y_probe = draw_wrapped(
        draw,
        body,
        (x + inner_pad, y + 68),
        FONT_SMALL,
        TITLE,
        w - (2 * inner_pad),
        line_gap=6,
    )
    h = max(148, body_y_probe - y + 22)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=22, fill=BOX, outline=LINE, width=2)
    draw.rounded_rectangle((x + 18, y + 18, x + 66, y + 62), radius=14, fill=BLUE_DARK)
    num = title.split(".")[0]
    num_w = draw.textlength(num, font=FONT_BODY_BOLD)
    draw.text((x + 42 - (num_w / 2), y + 25), num, font=FONT_BODY_BOLD, fill="white")
    draw.text((x + 82, y + 22), title[3:] if title[1] == "." else title, font=FONT_BODY_BOLD, fill=BLUE_DARK)
    draw_wrapped(draw, body, (x + inner_pad, y + 72), FONT_SMALL, TITLE, w - (2 * inner_pad), line_gap=6)
    return h


def make_page_1():
    img = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(img)
    y = MARGIN_Y

    draw.text((MARGIN_X, y), "Struktur Plan Penelitian", font=FONT_TITLE, fill=TITLE)
    y += text_height(FONT_TITLE) + 8
    draw.text(
        (MARGIN_X, y),
        "Prediksi Diabetes Mellitus Tipe 2 dengan Machine Learning dan Explainable AI",
        font=FONT_SUBTITLE,
        fill=SUB,
    )
    y += text_height(FONT_SUBTITLE) + 34

    intro = (
        "Dokumen ini merangkum alur kerja dari tahap prepare data sampai modeling dan interpretasi XAI, "
        "serta daftar uji coba yang perlu dilakukan agar eksperimen rapi, terukur, dan mudah dibandingkan."
    )
    y = draw_wrapped(draw, intro, (MARGIN_X, y), FONT_BODY, TITLE, CONTENT_W, line_gap=8)
    y += 26

    draw.text((MARGIN_X, y), "Alur Kerja Utama", font=FONT_H1, fill=BLUE_DARK)
    y += text_height(FONT_H1) + 20

    steps = [
        ("1. Audit & Integrasi", "Cek struktur data, join semua tabel ke patient, dan validasi label serta PatientGuid."),
        ("2. Prepare Data", "Tangani zero semu, missing values, encoding, scaling, dan pembagian train-test."),
        ("3. EDA Terarah", "Lihat imbalance, distribusi fitur, korelasi, sparsity medication, dan beda DM vs non-DM."),
        ("4. Feature Set", "Buat skenario core, clinical, dan full features agar eksperimen bisa dibandingkan."),
        ("5. Baseline Model", "Latih Logistic Regression, SVM, dan KNN sebagai pembanding awal."),
        ("6. Boosting Model", "Latih Gradient Boosting, XGBoost, dan LightGBM sebagai kandidat model utama."),
    ]

    col_gap = 28
    box_w = (CONTENT_W - col_gap) // 2
    left_x = MARGIN_X
    right_x = MARGIN_X + box_w + col_gap
    left_y = y
    right_y = y
    for idx, step in enumerate(steps):
        if idx % 2 == 0:
            h = step_box(draw, left_x, left_y, box_w, step[0], step[1])
            left_y += h + 22
        else:
            h = step_box(draw, right_x, right_y, box_w, step[0], step[1])
            right_y += h + 22

    footer_y = PAGE_H - MARGIN_Y + 12
    draw.line((MARGIN_X, footer_y - 24, PAGE_W - MARGIN_X, footer_y - 24), fill=LINE, width=2)
    draw.text((MARGIN_X, footer_y), "PracticeFusion Research Plan", font=FONT_SMALL, fill=SUB)
    draw.text((PAGE_W - MARGIN_X - 24, footer_y), "1", font=FONT_SMALL, fill=SUB)
    return img


def make_page_2():
    img = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(img)
    y = MARGIN_Y

    draw.text((MARGIN_X, y), "Lanjutan Plan dan Uji Coba", font=FONT_H1, fill=BLUE_DARK)
    y += text_height(FONT_H1) + 22

    top_boxes = [
        ("7. Tuning", "Optimasi hyperparameter dengan Optuna pada data train menggunakan cross-validation."),
        ("8. Evaluasi", "Ukur confusion matrix, accuracy, precision, recall, F1-score, ROC-AUC, dan PR-AUC."),
        ("9. XAI", "Jalankan SHAP pada model terbaik untuk global dan local explanation."),
        ("10. Kesimpulan", "Bandingkan hasil, cek leakage, dan rumuskan model terbaik serta keterbatasannya."),
    ]
    col_gap = 28
    box_w = (CONTENT_W - col_gap) // 2
    left_x = MARGIN_X
    right_x = MARGIN_X + box_w + col_gap
    left_y = y
    right_y = y
    for idx, step in enumerate(top_boxes):
        if idx % 2 == 0:
            h = step_box(draw, left_x, left_y, box_w, step[0], step[1])
            left_y += h + 18
        else:
            h = step_box(draw, right_x, right_y, box_w, step[0], step[1])
            right_y += h + 18

    y = max(left_y, right_y) + 18
    draw.text((MARGIN_X, y), "Uji Coba yang Wajib Dilakukan", font=FONT_H1, fill=BLUE_DARK)
    y += text_height(FONT_H1) + 18

    sections = [
        ("A. Data Preparation", [
            "Bandingkan data sebelum dan sesudah penanganan zero semu seperti BMI_Min = 0 dan Weight_Min = 0.",
            "Uji strategi imputasi sederhana dan konsisten untuk fitur numerik dan indikator.",
            "Bandingkan skenario dengan medication dan tanpa medication.",
        ]),
        ("B. Feature Set", [
            "Core features: patient + transcript.",
            "Clinical features: core + diagnosis + physician specialty.",
            "Full features: semua fitur termasuk medication.",
        ]),
        ("C. Model & Evaluasi", [
            "Bandingkan baseline vs boosting, default vs tuning Optuna, dan threshold default vs threshold optimum.",
            "Fokus utama pemilihan model: recall pasien diabetes, lalu F1-score dan PR-AUC.",
        ]),
        ("D. XAI", [
            "Gunakan SHAP untuk summary plot, dependence plot, dan contoh penjelasan pasien individual.",
        ]),
    ]

    for title, bullets in sections:
        draw.rounded_rectangle((MARGIN_X, y, PAGE_W - MARGIN_X, y + 50), radius=16, fill=BOX_ALT, outline=LINE)
        draw.text((MARGIN_X + 18, y + 10), title, font=FONT_H2, fill=BLUE_DARK)
        y += 66
        y = draw_bullet_list(draw, bullets, MARGIN_X + 6, y, CONTENT_W - 6)
        y += 10

    footer_y = PAGE_H - MARGIN_Y + 12
    draw.line((MARGIN_X, footer_y - 24, PAGE_W - MARGIN_X, footer_y - 24), fill=LINE, width=2)
    draw.text((MARGIN_X, footer_y), "PracticeFusion Research Plan", font=FONT_SMALL, fill=SUB)
    draw.text((PAGE_W - MARGIN_X - 24, footer_y), "2", font=FONT_SMALL, fill=SUB)
    return img


def make_page_3():
    img = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(img)
    y = MARGIN_Y

    draw.text((MARGIN_X, y), "Checklist Eksekusi Penelitian", font=FONT_H1, fill=BLUE_DARK)
    y += text_height(FONT_H1) + 18

    paragraph = (
        "Gunakan halaman ini sebagai ringkasan operasional saat eksperimen berjalan. "
        "Tujuannya supaya setiap tahap punya output yang jelas dan semua hasil bisa dilacak."
    )
    y = draw_wrapped(draw, paragraph, (MARGIN_X, y), FONT_BODY, TITLE, CONTENT_W, line_gap=8)
    y += 20

    draw.text((MARGIN_X, y), "Output Minimal yang Harus Ada", font=FONT_H2, fill=BLUE_DARK)
    y += text_height(FONT_H2) + 16
    y = draw_bullet_list(draw, [
        "Dataset final siap modeling.",
        "Tabel eksperimen yang seragam untuk semua model.",
        "Skor train CV dan test set untuk setiap skenario.",
        "Model terbaik final dengan hyperparameter terbaik.",
        "Visualisasi SHAP dan narasi interpretasi faktor penting.",
    ], MARGIN_X + 6, y, CONTENT_W - 6)

    y += 16
    draw.text((MARGIN_X, y), "Catatan Penting", font=FONT_H2, fill=BLUE_DARK)
    y += text_height(FONT_H2) + 16
    y = draw_bullet_list(draw, [
        "Waspadai leakage bila label diabetes terlalu dekat dengan fitur diagnosis atau medication.",
        "Simpan seed, split, dan parameter eksperimen agar hasil bisa direproduksi.",
        "Semua oversampling seperti SMOTE hanya boleh dilakukan di data train, bukan sebelum split.",
        "Kalau performa melonjak terlalu tinggi saat diagnosis atau medication masuk, lakukan sensitivity check.",
    ], MARGIN_X + 6, y, CONTENT_W - 6)

    y += 20
    draw.rounded_rectangle((MARGIN_X, y, PAGE_W - MARGIN_X, y + 220), radius=24, fill=BOX, outline=LINE, width=2)
    draw.text((MARGIN_X + 24, y + 20), "Urutan Praktis Eksekusi", font=FONT_H2, fill=BLUE_DARK)
    draw_wrapped(
        draw,
        "1. Finalkan cleaning dan join data. 2. Buat tiga versi feature set. "
        "3. Jalankan baseline. 4. Jalankan boosting. 5. Tuning Optuna. "
        "6. Bandingkan metrik. 7. Pilih model terbaik. 8. Jalankan SHAP dan tulis interpretasinya.",
        (MARGIN_X + 24, y + 66),
        FONT_BODY,
        TITLE,
        CONTENT_W - 48,
        line_gap=8,
    )

    footer_y = PAGE_H - MARGIN_Y + 12
    draw.line((MARGIN_X, footer_y - 24, PAGE_W - MARGIN_X, footer_y - 24), fill=LINE, width=2)
    draw.text((MARGIN_X, footer_y), "PracticeFusion Research Plan", font=FONT_SMALL, fill=SUB)
    draw.text((PAGE_W - MARGIN_X - 24, footer_y), "3", font=FONT_SMALL, fill=SUB)
    return img


def build_pdf():
    DOCS_DIR.mkdir(exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    pages = [make_page_1(), make_page_2(), make_page_3()]
    for idx, page in enumerate(pages, start=1):
        page.save(QA_DIR / f"page-{idx}.png")

    rgb_pages = [page.convert("RGB") for page in pages]
    rgb_pages[0].save(PDF_PATH, save_all=True, append_images=rgb_pages[1:], resolution=150.0)
    print(PDF_PATH)


if __name__ == "__main__":
    build_pdf()
