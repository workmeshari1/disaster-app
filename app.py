import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer, util
import torch

# --- إعداد الصفحة ---
st.set_page_config(page_title="⚡ إدارة الكوارث والأزمات", layout="centered")

# --- تحميل البيانات والموديل ---
@st.cache_data(ttl=600)
def load_data():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    # تحميل بيانات JSON من Secrets (بصيغة dict)
    creds_dict = st.secrets["GOOGLE_CREDENTIALS"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)

    # فتح الشيت
    sheet_id = st.secrets["SHEET_ID"]
    sheet = client.open_by_key(sheet_id)
    data_sheet = sheet.sheet1

    # قراءة البيانات
    data = data_sheet.get_all_records()
    df = pd.DataFrame(data)

    # كلمة المرور (إما من Secrets أو من الشيت نفسه)
    if "PASSWORD_CELL" in st.secrets:
        cell = st.secrets["PASSWORD_CELL"]  # مثل E1
        row = int(''.join(filter(str.isdigit, cell)))
        col_letters = ''.join(filter(str.isalpha, cell)).upper()
        col = sum((ord(c) - 64) * (26 ** i) for i, c in enumerate(reversed(col_letters)))
        password_cell = data_sheet.cell(row, col).value
    else:
        password_cell = data_sheet.cell(1, 5).value  # E1 افتراضياً

    # تجهيز الموديل + embeddings
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    descriptions = df['وصف الحالة أو الحدث'].fillna("").astype(str).tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=True)

    return df, model, embeddings, password_cell

# --- تحميل البيانات ---
df, model, embeddings, PASSWORD = load_data()

# --- التحقق من الجلسة ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h2>🔐 ادخل الرقم السري</h2>", unsafe_allow_html=True)
    password = st.text_input("الرقم السري", type="password")
    if st.button("دخول"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ الرقم السري غير صحيح")
else:
    # --- الواجهة الرئيسية ---
    st.markdown("<h2>⚡ دائرة إدارة الكوارث والأزمات الصناعية</h2>", unsafe_allow_html=True)

    query = st.text_input("ابحث هنا:")

    if query:
        # البحث الذكي (semantic)
        query_embedding = model.encode(query, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cosine_scores, k=2)

        st.markdown("### 🔎 نتائج البحث الذكي:")
        for score, idx in zip(top_results[0], top_results[1]):
            r = df.iloc[idx.item()]
            st.markdown(
                f"""
                <div style='background-color:#222;color:white;padding:10px;border-radius:5px;direction:rtl;text-align:right;font-size:18px;'>
                <b>الوصف:</b> {r['وصف الحالة أو الحدث']}<br>
                <b>الإجراء:</b>
                <span style='background-color:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:5px;display:inline-block;'>
                {r['الإجراء']}
                </span><br>
                <span style='font-size:14px;color:orange;'>درجة التشابه: {score:.2f}</span>
                </div><br>
                """,
                unsafe_allow_html=True,
            )

    if st.button("🔒 تسجيل خروج"):
        st.session_state.authenticated = False
        st.rerun()
