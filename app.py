import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import json

# ---------- إعداد الصفحة ----------
st.set_page_config(page_title="⚡ إدارة الكوارث والأزمات", layout="centered")

# ---------- CSS للواجهة ----------
st.markdown("""
<style>
html, body, [class*="css"] { direction: rtl; }
h2 { color: #ff6600 !important; text-align: center; }
.box { background:#1f1f1f; color:white; padding:10px; border-radius:8px; }
.pill { background:#ff6600; color:#0a1e3f; padding:4px 8px; border-radius:6px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ---------- إعداد Google Sheet ----------
SHEET_ID = "11BWnvPjcRZwnGhynCCyYCc7MGfHJlSyJCqwHI6z4KJI"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ---------- تحميل الموديل ----------
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2')

# ---------- تحميل البيانات من Google Sheet ----------
@st.cache_data(ttl=600, show_spinner="جارِ تحميل البيانات من Google Sheet ...")
def load_data():
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        st.error("لم يتم العثور على GOOGLE_CREDENTIALS في Secrets.")
        st.stop()

    # قراءة JSON من Secrets
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)

    try:
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.sheet1
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        # كلمة السر من E1
        password = worksheet.cell(1, 5).value

        # تجهيز الموديل
        model = load_model()
        descriptions = df["وصف الحالة أو الحدث"].fillna("").astype(str).tolist()
        embeddings = model.encode(descriptions, convert_to_tensor=True)

        return df, model, embeddings, password
    except Exception as e:
        st.error(f"❌ فشل الاتصال بجوجل شيت: {e}")
        st.stop()

# ---------- تحميل البيانات ----------
df, model, embeddings, PASSWORD = load_data()

# ---------- زر تحديث البيانات ----------
if st.button("🔄 تحديث البيانات من Google Sheet"):
    load_data.clear()
    st.success("تم تحديث البيانات.")
    st.rerun()

# ---------- المصادقة ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h2>ادخل الرقم السري</h2>", unsafe_allow_html=True)
    password = st.text_input("الرقم السري", type="password")
    if st.button("دخول") or st.session_state.get("enter_pressed", False):
        if (PASSWORD or "") == (password or ""):
            st.session_state.authenticated = True
            st.session_state.enter_pressed = False
            st.rerun()
        else:
            st.error("❌ الرقم السري غير صحيح")
else:
    st.markdown("<h2>⚡ دائرة إدارة الكوارث والأزمات الصناعية</h2>", unsafe_allow_html=True)

    query = st.text_input("ابحث هنا:", placeholder="مثال: حريق في غرفة المولد الاحتياطية")

    if query:
        words = [w for w in query.lower().split() if w]

        literal_results = []
        literal_indices = set()
        synonym_results = []

        # البحث الحرفي
        for idx, row in df.iterrows():
            text = str(row["وصف الحالة أو الحدث"]).lower()
            if all(word in text for word in words):
                literal_results.append((idx, row))
                literal_indices.add(idx)

        # البحث بالمرادفات
        for idx, row in df.iterrows():
            if idx in literal_indices:
                continue
            text = str(row["وصف الحالة أو الحدث"]).lower()
            synonyms = str(row.get("مرادفات للوصف", "") or "").lower().split(",")
            synonyms = [s.strip() for s in synonyms if s.strip()]
            matched = any((word in text) or any(word in s for s in synonyms) for word in words)
            if matched:
                synonym_results.append((idx, row))

        # عرض نتائج البحث الحرفي
        if literal_results:
            st.write("### النتائج المطابقة حرفيًا")
            for _, r in literal_results[:2]:
                st.markdown(f"<div class='box'><b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br><b>الإجراء:</b> <span class='pill'>{r['الإجراء']}</span></div><br>", unsafe_allow_html=True)

        # عرض نتائج المرادفات
        if synonym_results:
            st.write("### 👀 قد تقصد أيضًا")
            for _, r in synonym_results:
                st.markdown(f"<div class='box' style='background:#333;'><b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br><b>الإجراء:</b> <span class='pill'>{r['الإجراء']}</span></div><br>", unsafe_allow_html=True)

        # البحث الدلالي
        if st.button("🔍 جرب البحث الذكي (Similarity)"):
            query_embedding = model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            topk = min(2, len(df))
            top_results = torch.topk(cosine_scores, k=topk)

            st.write("### 🔎 نتائج البحث الذكي")
            for score, idx in zip(top_results[0], top_results[1]):
                r = df.iloc[int(idx)]
                st.markdown(f"<div class='box' style='background:#444;'><b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br><b>الإجراء:</b> <span class='pill'>{r['الإجراء']}</span><br><span style='font-size:14px;color:orange;'>درجة التشابه: {float(score):.2f}</span></div><br>", unsafe_allow_html=True)

        st.info("إذا ما وصلك الإجراء الصحيح: جرّب كلمات أكثر دقة أو استخدم البحث الذكي.")

    if st.button("🔒 تسجيل خروج"):
        st.session_state.authenticated = False
        st.rerun()
