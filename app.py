import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# إعداد الصفحة
st.set_page_config(page_title="⚡ إدارة الكوارث والأزمات", layout="centered", initial_sidebar_state="collapsed")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# تحميل النموذج مرة واحدة (نموذج محسن للعربية)
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-large")

# تحميل البيانات وكلمة المرور
@st.cache_data(ttl=600)
def load_data_and_password():
    try:
        creds_info = dict(st.secrets["GOOGLE_CREDENTIALS"])
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(st.secrets["SHEET"]["id"]).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        password = sheet.cell(1, 5).value or ""
        return df, password
    except Exception as e:
        st.error(f"❌ فشل تحميل البيانات: {str(e)}")
        return None, None

# حساب إمبادنج للوصف
@st.cache_data
def compute_embeddings(descriptions: list[str]):
    model = load_model()
    return model.encode(descriptions, convert_to_tensor=True, show_progress_bar=False)

# عرض بطاقة النتيجة
def render_card(row, icon="🔶", score=None):
    score_text = f"<br><span style='font-size:14px;color:orange;'>درجة التشابه: {float(score):.2f}</span>" if score is not None else ""
    st.markdown(
        f"""
        <div style='background:#1f1f1f;color:#fff;padding:12px;border-radius:8px;direction:rtl;text-align:right;font-size:18px;margin-bottom:10px;'>
            <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
            <b>الوصف:</b> {row['وصف الحالة أو الحدث']}<br>
            <b>الإجراء:</b>
            <span style='background:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:6px;display:inline-block;margin-top:4px;'>
                {row['الإجراء']}
            </span>{score_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# الواجهة
st.title("⚡ دائرة إدارة الكوارث والأزمات الصناعية")

# تحميل البيانات
df, PASSWORD = load_data_and_password()
if df is None or PASSWORD is None:
    st.stop()

# التحقق من الأعمدة
DESC_COL, ACTION_COL, SYN_COL = "وصف الحالة أو الحدث", "الإجراء", "مرادفات للوصف"
required_cols = [DESC_COL, ACTION_COL]
for col in required_cols:
    if col not in df.columns:
        st.error(f"عمود مفقود في Google Sheet: '{col}'")
        st.stop()
df[SYN_COL] = df.get(SYN_COL, "").fillna("")

# تسجيل الدخول باستخدام form لدعم Enter
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("ادخل الرقم السري")
    with st.form(key="login_form"):
        password = st.text_input("الرقم السري", type="password")
        submit = st.form_submit_button("دخول")
        if submit:
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ الرقم السري غير صحيح")
    st.stop()

# البحث
query = st.text_input("ابحث هنا:", placeholder="اكتب وصف الحالة…")
if not query.strip():
    st.info("⚡ 🔥 🚔 🚗 🛢️ 💧")
    st.stop()

# معالجة الاستعلام
query_lower = query.strip().lower()
words = query_lower.split()

# البحث الحرفي (وصف + مرادفات)
literal_results = []
for _, row in df.iterrows():
    desc_text = str(row[DESC_COL]).lower()
    syn_text = str(row[SYN_COL]).lower()
    combined_text = desc_text + " " + syn_text
    if all(w in combined_text for w in words):
        literal_results.append(row)

# حساب النتائج المعنوية (دلالية) دائمًا
descriptions = df[DESC_COL].fillna("").astype(str).tolist()
embeddings = compute_embeddings(descriptions)
query_embedding = load_model().encode(query, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
top_scores, top_indices = torch.topk(cosine_scores, k=min(5, len(df)))  # أكثر نتائج للاختيار

semantic_results = []
threshold = 0.5  # عتبة للتشابه لضمان الدقة
literal_indices = [df.index[df[DESC_COL] == r[DESC_COL]].tolist()[0] for r in literal_results] if literal_results else []
for score, idx in zip(top_scores, top_indices):
    if score > threshold and int(idx.item()) not in literal_indices:  # تجنب التكرار مع الحرفية
        semantic_results.append((df.iloc[int(idx.item())], score))

# عرض النتائج
if literal_results:
    st.subheader("🔍 النتائج الحرفية:")
    for r in literal_results[:3]:
        render_card(r, "🔍")

# عرض النتائج المعنوية دائمًا تحت "يمكن تقصد"
if semantic_results:
    st.subheader("يمكن تقصد🧐")
    for r, score in semantic_results[:3]:
        render_card(r, "🤖", score)
elif not literal_results:
    st.warning("❌ لم يتم العثور على نتائج حرفية أو معنوية ذات صلة. جرب صياغة أخرى!")

# تسجيل الخروج
if st.button("🔒 تسجيل خروج"):
    st.session_state.authenticated = False
    st.rerun()
