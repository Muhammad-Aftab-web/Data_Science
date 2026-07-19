import os
import streamlit as st

import config
from authentication import auth
from database import db
from utils.helpers import inject_theme_css, toast
from machine_learning.preprocessing import load_dataset
from machine_learning.training import train_and_compare, save_best_model, set_active_model

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
auth.require_admin()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("⚙️ Administrator Settings")

tab_users, tab_models, tab_data, tab_db = st.tabs(
    ["👥 Users", "🤖 Models", "📂 Dataset", "🗄️ Database"]
)

# ---------------------------------------------------------------------------
with tab_users:
    st.subheader("All Users")
    users = db.get_all_users()
    st.dataframe(
        [{"ID": u.user_id, "Name": u.full_name, "Username": u.username,
          "Email": u.email, "Role": u.role, "Joined": u.created_at} for u in users],
        use_container_width=True, hide_index=True,
    )
    del_id = st.number_input("User ID to delete", min_value=0, value=0, step=1)
    if st.button("🗑️ Delete User") and del_id:
        if del_id == st.session_state["auth_user_id"]:
            st.error("You cannot delete your own account while logged in.")
        else:
            db.delete_user(del_id)
            toast("User deleted.")
            st.rerun()

# ---------------------------------------------------------------------------
with tab_models:
    st.subheader("Retrain Model")
    if os.path.exists(config.PROCESSED_DATASET_PATH):
        if st.button("🔁 Retrain Now", type="primary"):
            df = load_dataset(config.PROCESSED_DATASET_PATH)
            progress = st.progress(0.0)
            def _cb(i, total, name):
                progress.progress(i / total, text=f"Training {name}...")
            results, best_name, encoders, _ = train_and_compare(df, progress_callback=_cb)
            save_best_model(results, best_name)
            toast(f"Retrained. Best model: {best_name}")
    else:
        st.info("No processed dataset found. Upload one in Dataset Management first.")

    st.subheader("Change Active Model")
    if os.path.exists(config.BEST_MODEL_META_PATH):
        import joblib
        meta = joblib.load(config.BEST_MODEL_META_PATH)
        options = list(meta["all_model_paths"].keys())
        current = meta["best_model_name"]
        choice = st.selectbox("Active model", options, index=options.index(current))
        if st.button("Set as Active Model"):
            set_active_model(choice)
            toast(f"Active model set to {choice}.")
    else:
        st.info("No trained models yet.")

# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Upload New Dataset")
    uploaded = st.file_uploader("Replace dataset", type=["csv"], key="settings_upload")
    if uploaded is not None:
        df = load_dataset(uploaded)
        df.to_csv(config.RAW_DATASET_PATH, index=False)
        toast(f"New dataset saved ({len(df)} rows). Go to Dataset Management to preprocess it.")

    st.subheader("Reset Prediction History")
    st.warning("This permanently deletes ALL predictions from ALL users.")
    if st.button("🧨 Reset All Prediction History"):
        db.reset_prediction_history()
        toast("Prediction history has been reset.")

# ---------------------------------------------------------------------------
with tab_db:
    st.subheader("Backup Database")
    if st.button("💾 Create Backup"):
        try:
            path = db.backup_database()
            toast(f"Backup created: {os.path.basename(path)}")
        except RuntimeError as e:
            st.error(str(e))

    st.subheader("Restore Database")
    backups = db.list_backups()
    if backups:
        chosen = st.selectbox("Choose a backup to restore", backups)
        if st.button("♻️ Restore Selected Backup"):
            db.restore_database(os.path.join(config.BACKUP_DIR, chosen))
            toast("Database restored. Please refresh the app.")
    else:
        st.info("No backups found yet.")
