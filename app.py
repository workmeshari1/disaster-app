import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import json, os, re

# ---------- الإعدادات ----------
SHEET_ID = "11BWnvPjcRZwnGhynCCyYCc7MGfHJlSyJCqwHI6z4KJI"

# ---------- تنظيف النصوص العربية ----------
def normalize_arabic(text):
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"[إأآا]", "ا", text)  # توحيد الألف
    text = re.sub(r"ى", "ي", text)        # توحيد الياء
    return text.strip()

# ---------- الاتصال بـ Google Sheet ----------
@st.cache_data(ttl=600)
def load_data():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        service_account_info = json.loads(os.getenv("GCP_SERVICE_ACCOUNT"))
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        data_sheet = sheet.sheet1

        # قراءة البيانات
        data = data_sheet.get_all_records()
        df = pd.DataFrame(data)

        # كلمة المرور من E1
        password_cell = data_sheet.cell(1, 5).value  # E1

        # تجهيز الموديل (LaBSE)
        model = SentenceTransformer("sentence-transformers/LaBSE")

        # تجهيز المتجهات مع تنظيف النصوص
        descriptions = [normalize_arabic(str(d)) for d in df['وصف الحالة أو الحدث'].fillna("")]
        embeddings = model.encode(descriptions, convert_to_tensor=True)

        return df, model, embeddings, password_cell
    except Exception as e:
        st.error(f"❌ فشل الاتصال بجوجل شيت: {e}")
        st.stop()

# تحميل البيانات
df, model, embeddings, PASSWORD = load_data()

# ---------- واجهة ----------
st.set_page_config(page_title="⚡ إدارة الكوارث والأزمات", layout="centered", initial_sidebar_state="collapsed")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h2>ادخل الرقم السري</h2>", unsafe_allow_html=True)
    password = st.text_input("الرقم السري", type="password")
    if st.button("دخول") or st.session_state.get("enter_pressed", False):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.session_state.enter_pressed = False
            st.rerun()
        else:
            st.error("❌ الرقم السري غير صحيح")
else:
    st.markdown("<h2>⚡ دائرة إدارة الكوارث والأزمات الصناعية</h2>", unsafe_allow_html=True)

    # مربع البحث مع الرموز
    query = st.text_input("ابحث هنا:", placeholder="⚡ 🔥 🚔 🚗 🛢️ 💧")

    # عرض تلميح بالرموز
    st.info("⚡ مشاكل كهربائية  |  🔥 حرائق  |  🚔 أمن صناعي  |  🚗 حوادث سيارات  |  🛢️ برميل زيت  |  💧 انسكاب")

    if query:
        query_norm = normalize_arabic(query)
        words = query_norm.lower().split()
        literal_results = []
        synonym_results = []

        # --- 1) البحث الحرفي من الوصف ---
        for idx, row in df.iterrows():
            text = normalize_arabic(str(row['وصف الحالة أو الحدث']).lower())
            if all(word in text for word in words):
                literal_results.append(row)

        # --- 2) البحث الحرفي من المرادفات ---
        if not literal_results:  # فقط لو مافيه نتائج من الوصف
            for idx, row in df.iterrows():
                synonyms = normalize_arabic(str(row.get('مرادفات للوصف', '')).lower()).split(',')
                synonyms = [s.strip() for s in synonyms if s.strip()]
                if any(word in synonyms for word in words):
                    synonym_results.append(row)

        # --- عرض النتائج ---
        if literal_results:
            st.subheader("🔍 نتائج البحث الحرفي (من الوصف):")
            for r in literal_results[:3]:
                st.markdown(
                    f"""
                    <div style='background-color:#1f1f1f;color:white;padding:10px;border-radius:5px;direction:rtl;text-align:right;font-size:18px;'>
                        <b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br>
                        <b>الإجراء:</b>
                        <span style='background-color:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:5px;'>
                        {r['الإجراء']}
                        </span>
                    </div><br>
                    """, unsafe_allow_html=True)

        elif synonym_results:
            st.subheader("📌 نتائج من المرادفات:")
            for r in synonym_results[:3]:
                st.markdown(
                    f"""
                    <div style='background-color:#333;color:white;padding:10px;border-radius:5px;direction:rtl;text-align:right;font-size:18px;'>
                        <b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br>
                        <b>الإجراء:</b>
                        <span style='background-color:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:5px;'>
                        {r['الإجراء']}
                        </span>
                    </div><br>
                    """, unsafe_allow_html=True)

        else:
            # --- 3) البحث الذكي ---
            st.warning("❌ لم يتم العثور على نتائج حرفية .. جرب البحث الذكي 👇")

            if st.button("🤖 جرب البحث الذكي"):
                query_embedding = model.encode(query_norm, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
                top_results = torch.topk(cosine_scores, k=3)

                st.subheader("🔎 نتائج البحث الذكي:")
                for score, idx in zip(top_results[0], top_results[1]):
                    r = df.iloc[idx.item()]
                    st.markdown(
                        f"""
                        <div style='background-color:#444;color:white;padding:10px;border-radius:5px;direction:rtl;text-align:right;font-size:18px;'>
                            <b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br>
                            <b>الإجراء:</b>
                            <span style='background-color:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:5px;'>
                            {r['الإجراء']}
                            </span><br>
                            <span style='font-size:14px;color:orange;'>درجة التشابه: {score:.2f}</span>
                        </div><br>
                        """, unsafe_allow_html=True)

    if st.button("🔒 تسجيل خروج"):
        st.session_state.authenticated = False
        st.rerun()
