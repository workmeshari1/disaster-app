import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# ---------- الإعدادات ----------
SHEET_ID = "11BWnvPjcRZwnGhynCCyYCc7MGfHJlSyJCqwHI6z4KJI"
JSON_PATH = r"C:\Users\Meshari\smart-google-sheet\perfect-entry-469221-e3-2fb34a3b26e3.json"

# ---------- الاتصال بـ Google Sheet ----------
@st.cache_data(ttl=600)
def load_data():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(JSON_PATH, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        data_sheet = sheet.sheet1

        # قراءة البيانات
        data = data_sheet.get_all_records()
        df = pd.DataFrame(data)

        # قراءة كلمة المرور من الخلية E1 (العمود الخامس، الصف الأول)
        password_cell = data_sheet.cell(1, 5).value  # E1

        # تجهيز الموديل
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        descriptions = df['وصف الحالة أو الحدث'].tolist()
        embeddings = model.encode(descriptions, convert_to_tensor=True)

        return df, model, embeddings, password_cell
    except Exception as e:
        st.error(f"❌ فشل الاتصال بجوجل شيت: {e}")
        st.stop()

# تحميل البيانات
df, model, embeddings, PASSWORD = load_data()

# ---------- تعديل الخلفية والعناصر ----------
st.markdown("""
    <style>
    body {
        background-color: #0a1e3f;
        color: white;
        direction: rtl;
    }
    h2 {
        color: #ff6600 !important;
        text-align: center;
    }
    .action-box {
        background-color: #ff6600;
        color: #0a1e3f;
        padding: 10px;
        border-radius: 6px;
        font-size: 18px;
        margin-bottom: 12px;
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- واجهة Streamlit ----------
st.set_page_config(page_title="⚡ إدارة الكوارث والأزمات", layout="centered")

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

    query = st.text_input("ابحث هنا:")

    if query:
        query_lower = query.lower()
        words = query_lower.split()
        literal_results = []
        synonym_results = []

        # البحث الحرفي
        for idx, row in df.iterrows():
            text = str(row['وصف الحالة أو الحدث']).lower()
            if all(word in text for word in words):
                literal_results.append(row)

        # البحث بالمرادفات
        for idx, row in df.iterrows():
            text = str(row['وصف الحالة أو الحدث']).lower()
            synonyms = str(row.get('مرادفات للوصف', '')).lower().split(',')
            synonyms = [s.strip() for s in synonyms if s.strip()]
            if idx not in [r.name for r in literal_results]:
                if any(word in text or word in synonyms for word in words):
                    synonym_results.append(row)

        # عرض نتائج البحث الحرفي (أقرب 2 فقط)
        if literal_results:
            for r in literal_results[:2]:
                st.markdown(
                    f"""
                    <div style='background-color:#1f1f1f;color:white;padding:10px;border-radius:5px;direction:rtl;text-align:right;font-size:18px;display:flex;align-items:center;gap:10px;'>
                    <span style="font-size:24px;">🔍</span>
                    <div>
                        <b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br>
                        <b>الإجراء:</b>
                        <span style='background-color:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:5px;display:inline-block;'>
                        {r['الإجراء']}
                        </span><br>
                    </div>
                    </div><br>
                    """,
                    unsafe_allow_html=True,
                )

        # عرض نتائج المرادفات
        if synonym_results:
            st.markdown("👀 يمكن تقصد")
            for r in synonym_results:
                st.markdown(
                    f"""
                    <div style='background-color:#333;color:white;padding:10px;border-radius:5px;direction:rtl;text-align:right;font-size:18px;'>
                    <b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br>
                    <b>الإجراء:</b>
                    <span style='background-color:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:5px;display:inline-block;'>
                    {r['الإجراء']}
                    </span><br>
                    </div><br>
                    """,
                    unsafe_allow_html=True,
                )

        # البحث الذكي (أقرب نتيجتين فقط)
        if st.button("🔍 جرب البحث الذكي"):
            query_embedding = model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            top_results = torch.topk(cosine_scores, k=2)

            st.markdown("<b>🔎 نتائج البحث الذكي:</b>", unsafe_allow_html=True)
            for score, idx in zip(top_results[0], top_results[1]):
                r = df.iloc[idx.item()]
                st.markdown(
                    f"""
                    <div style='background-color:#444;color:white;padding:10px;border-radius:5px;direction:rtl;text-align:right;font-size:18px;display:flex;align-items:center;gap:10px;'>
                    <span style="font-size:24px;">🤖</span>
                    <div>
                        <b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br>
                        <b>الإجراء:</b>
                        <span style='background-color:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:5px;display:inline-block;'>
                        {r['الإجراء']}
                        </span><br>
                        <span style='font-size:14px;color:orange;'>درجة التشابه: {score:.2f}</span>
                    </div>
                    </div><br>
                    """,
                    unsafe_allow_html=True,
                )

        st.info("إذا ماوصلك الإجراء الصحيح ساعدني بوصف كلمات أكثر أو استخدم البحث الذكي")

    if st.button("🔒 تسجيل خروج"):
        st.session_state.authenticated = False
        st.rerun()
