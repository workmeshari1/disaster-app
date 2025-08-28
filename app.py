import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

# --- إعداد الصفحة ---
st.set_page_config(
    page_title="⚡ إدارة الكوارث والأزمات", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# --- الخلفية responsive ---
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("https://github.com/workmeshari1/disaster-app/blob/bb4be38238ac06288848fa086e098f56b21e92b4/assets.png?raw=true");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
@media only screen and (max-width: 768px) {{
    .stApp {{
        background-size: contain;
        background-position: top center;
    }}
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- إخفاء شعار Streamlit والفوتر والأيقونات الصغيرة ---
hide_st_style = """
<style>
footer {visibility: hidden;}
.stDeployButton {display:none;}
button[kind="header"] {display: none;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- إعداد Google Sheets و تحميل الموديل ---
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

@st.cache_resource
def load_model():
    return SentenceTransformer("Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka")

@st.cache_data(ttl=600)
def load_data_and_password():
    try:
        if hasattr(st, 'secrets') and "GOOGLE_CREDENTIALS" in st.secrets:
            creds_info = dict(st.secrets["GOOGLE_CREDENTIALS"])
        else:
            import json
            creds_json = os.getenv("GOOGLE_CREDENTIALS", "{}")
            creds_info = json.loads(creds_json)
        
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        client = gspread.authorize(creds)

        if hasattr(st, 'secrets') and "SHEET" in st.secrets:
            sheet_id = st.secrets["SHEET"]["id"]
        else:
            sheet_id = os.getenv("SHEET_ID", "")
        
        sheet = client.open_by_key(sheet_id)
        ws = sheet.sheet1

        data = ws.get_all_records()
        df = pd.DataFrame(data)
        password_value = ws.cell(1, 5).value
        return df, password_value
    except Exception as e:
        raise Exception(f"Failed to connect to Google Sheets: {str(e)}")

@st.cache_data
def compute_embeddings(descriptions: list[str]):
    model = load_model()
    return model.encode(descriptions, convert_to_tensor=True)

# ============== واجهة ==============
st.title("⚡ دائرة إدارة الكوارث والأزمات الصناعية")

try:
    df, PASSWORD = load_data_and_password()
except Exception as e:
    st.error(f"❌ فشل الاتصال بقاعدة البيانات: {str(e)}")
    st.info("تأكد من إعداد متغيرات البيئة أو أسرار Streamlit بشكل صحيح.")
    st.stop()

DESC_COL = "وصف الحالة أو الحدث"
ACTION_COL = "الإجراء"
SYN_COL = "مرادفات للوصف"

if df.empty:
    st.error("❌ لا توجد بيانات في الجدول.")
    st.stop()

for col in [DESC_COL, ACTION_COL]:
    if col not in df.columns:
        st.error(f"عمود مفقود: '{col}'")
        st.stop()

if SYN_COL not in df.columns:
    df[SYN_COL] = ""

# --- تسجيل الدخول ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("ادخل الرقم السري")
    
    # نص الإدخال للرقم السري
    password = st.text_input("الرقم السري", type="password", key="pass_input")

    # تحقق من Enter أو زر دخول
    def check_password():
        if st.session_state.pass_input == str(PASSWORD):
            st.session_state.authenticated = True
            st.session_state.pass_input = ""
            st.experimental_rerun()
        elif st.session_state.pass_input != "":
            st.error("❌ الرقم السري غير صحيح")

    # عندما يضغط Enter (على نفس الـ text_input)
    if st.session_state.pass_input != "":
        check_password()

    # زر دخول اختياري
    if st.button("دخول"):
        check_password()
        
    st.stop()

# --- بعد التحقق ---
query = st.text_input("ابحث هنا:", placeholder="اكتب وصف الحالة…")
if not query:
    st.info("⚡ 🔥 🚔 🚗 🛢️ 💧")
    st.stop()

q = query.strip().lower()
words = [w for w in q.split() if w]

literal_results = []
synonym_results = []

for _, row in df.iterrows():
    text = str(row[DESC_COL]).lower()
    if all(w in text for w in words):
        literal_results.append(row)

if not literal_results:
    for _, row in df.iterrows():
        syn_text = str(row.get(SYN_COL, "")).lower()
        if any(w in syn_text for w in words):
            synonym_results.append(row)

def render_card(r, icon="🔶"):
    st.markdown(
        f"""
        <div style='background:#1f1f1f;color:#fff;padding:12px;border-radius:8px;direction:rtl;text-align:right;font-size:18px;margin-bottom:10px;'>
            <div style="font-size:22px;margin-bottom:6px;">{icon} </div>
            <b>الوصف:</b> {r[DESC_COL]}<br>
            <b>الإجراء:</b>
            <span style='background:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:6px;display:inline-block;margin-top:4px;'>
                {r[ACTION_COL]}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if literal_results:
    st.subheader("🔍 النتائج المطابقة:")
    for r in literal_results[:3]:
        render_card(r, "🔍")
elif synonym_results:
    st.subheader("📌 يمكن قصدك:")
    for r in synonym_results[:3]:
        render_card(r, "📌")
else:
    st.warning("❌ لم يتم العثور على نتائج..")
    if st.button("🤖 اسأل الذكي"):
        with st.spinner("جاري البحث الذكي..."):
            model = load_model()
            descriptions = df[DESC_COL].fillna("").astype(str).tolist()
            embeddings = compute_embeddings(descriptions)
            query_embedding = model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            top_scores, top_indices = torch.topk(cosine_scores, k=min(3, len(df)))
            st.subheader("🧐 يمكن قصدك:")
            for score, idx in zip(top_scores, top_indices):
                if float(score) > 0.3:
                    r = df.iloc[int(idx.item())]
                    st.markdown(
                        f"""
                        <div style='background:#444;color:#fff;padding:12px;border-radius:8px;direction:rtl;text-align:right;font-size:18px;margin-bottom:10px;'>
                            <div style="font-size:22px;margin-bottom:6px;">🤖 </div>
                            <b>الوصف:</b> {r[DESC_COL]}<br>
                            <b>الإجراء:</b>
                            <span style='background:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:6px;display:inline-block;margin-top:4px;'>
                                {r[ACTION_COL]}
                            </span><br>
                            <span style='font-size:14px;color:orange;'>درجة التشابه: {float(score):.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

with st.sidebar:
    st.markdown("### معلومات النظام")
    st.info(f"📊 عدد الحالات المسجلة: {len(df)}")
    st.info("🔄 تحديث البيانات: كل 10 دقائق")
    
    if st.button("🔒 تسجيل خروج"):
        st.session_state.authenticated = False
        st.experimental_rerun()

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; direction: rtl;'>آلية إدارة الكوارث والأزمات الذكية</div>",
    unsafe_allow_html=True
)
