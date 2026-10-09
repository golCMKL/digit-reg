import os
from pathlib import Path

import psycopg
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Digit Labeling")
st.title("Digit Image Labeling")
st.write("Inspect each image and select its actual digit.")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "digit_db"),
    "user": os.getenv("DB_USER", "digit_user"),
    "password": os.getenv("DB_PASSWORD", "digit_password"),
}

# Project root is the parent of retraining/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_unlabeled_images():
    with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute("""
                SELECT p.id, p.prediction, p.confidence, p.image_path
                FROM predictions AS p
                LEFT JOIN labeled_samples AS l
                    ON l.prediction_id = p.id
                WHERE l.id IS NULL
                  AND p.image_path IS NOT NULL
                  AND p.error IS NULL
                ORDER BY p.id DESC
            """)
        return cur.fetchall()


def save_label(prediction_id, image_path, true_label):
    with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute("""
                INSERT INTO labeled_samples
                    (prediction_id, image_path, true_label)
                VALUES (%s, %s, %s)
                ON CONFLICT (prediction_id) DO NOTHING
            """, (prediction_id, image_path, true_label))


try:
    samples = get_unlabeled_images()
except psycopg.Error as exc:
    st.error(f"Could not load samples from PostgreSQL: {exc}")
    st.stop()

if not samples:
    st.success("No unlabeled images found.")
    st.stop()

st.caption(f"{len(samples)} image(s) waiting for labels.")

# Choose an image to review.
sample_options = {
    f"ID {row[0]} — model predicted {row[1]} "
    f"(confidence {row[2]:.3f})": row
    for row in samples
}
selected_name = st.selectbox(
    "Choose an image",
    list(sample_options.keys()),
)
sample = sample_options[selected_name]

prediction_id, prediction, confidence, image_path = sample

# The stored path is relative to the project root.
image_file = PROJECT_ROOT / image_path

if not image_file.is_file():
    st.error(f"Image file not found: {image_file}")
    st.stop()

st.image(
    Image.open(image_file).convert("L"),
    caption=f"Prediction ID {prediction_id}",
    width=280,
)

st.write(f"**Model prediction:** {prediction}")
st.write(f"**Confidence:** {confidence:.4f}")

with st.form("label_form"):
    true_label = st.selectbox(
        "What digit is actually shown?",
        options=list(range(10)),
        index=0,
    )
    submitted = st.form_submit_button("Save verified label")

if submitted:
    try:
        save_label(prediction_id, image_path, true_label)
        st.success(
            f"Saved label {true_label} for prediction {prediction_id}."
        )
        st.rerun()
    except psycopg.Error as exc:
        st.error(f"Could not save label: {exc}")
    else:
        st.rerun()