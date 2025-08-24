
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="⚡ إدارة الكوارث والأزمات", layout="centered", initial_sidebar_state="collapsed")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# --- تحميل الموديل مرة واحدة ---
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/LaBSE")


# --- قراءة البيانات + كلمة المرور من الشيت (كل 10 دق) ---
@st.cache_data(ttl=600)
def load_data_and_password():
    # نقرأ الدكت مباشرة من st.secrets (بدون json.loads)
    creds_info = dict(st.secrets["GOOGLE_CREDENTIALS"])
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(st.secrets["SHEET"]["id"])
    ws = sheet.sheet1

    data = ws.get_all_records()
    df = pd.DataFrame(data)

    # كلمة المرور من E1 (صف 1 عمود 5)
    password_value = ws.cell(1, 5).value

    return df, password_value

# --- حساب إمبادنج للوصف (يتحدّث فقط عند تغيّر البيانات) ---
@st.cache_data
def compute_embeddings(descriptions: list[str]):
    model = load_model()
    return model.encode(descriptions, convert_to_tensor=True)

# ============== واجهة ==============
st.title("⚡ دائرة إدارة الكوارث والأزمات الصناعية")

# جرّب تحميل البيانات
try:
    df, PASSWORD = load_data_and_password()
except Exception as e:
    st.error("❌ فشل الاتصال. (service account).")
    st.stop()

# التحقق من الأعمدة المطلوبة
DESC_COL = "وصف الحالة أو الحدث"
ACTION_COL = "الإجراء"
SYN_COL = "مرادفات للوصف"
for col in [DESC_COL, ACTION_COL]:
    if col not in df.columns:
        st.error(f"عمود مفقود في Google Sheet: '{col}'. تأكد من اسم العمود حرفيًا.")
        st.stop()
if SYN_COL not in df.columns:
    # لو ناقص، نضيفه عمود فاضي لتجنّب الأخطاء
    df[SYN_COL] = ""

# تسجيل الدخول
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("ادخل الرقم السري")
    password = st.text_input("الرقم السري", type="password")
    if st.button("دخول"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ الرقم السري غير صحيح")
    st.stop()

# بعد التحقق
query = st.text_input("ابحث هنا:", placeholder="اكتب وصف الحالة…")
if not query:
    st.info("⚡ 🔥 🚔 🚗 🛢️ 💧")
    st.stop()

# ---------- البحث الحرفي ----------
q = query.strip().lower()
words = [w for w in q.split() if w]

literal_results = []
synonym_results = []

# 1) الحرفي من الوصف
for _, row in df.iterrows():
    text = str(row[DESC_COL]).lower()
    if all(w in text for w in words):
        literal_results.append(row)

# 2) الحرفي من المرادفات (نبحث داخل النص كاملًا وليس مساواة تامة)
if not literal_results:
    for _, row in df.iterrows():
        syn_text = str(row.get(SYN_COL, "")).lower()
        # نعتبر أي كلمة من كلمات البحث موجودة ضمن المرادفات
        if any(w in syn_text for w in words):
            synonym_results.append(row)

# عرض أقرب 3 نتائج من كل نوع
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
    st.subheader("🔍:")
    for r in literal_results[:3]:
        render_card(r, "🔍")
elif synonym_results:
    st.subheader("📌 يمكن قصدك:")
    for r in synonym_results[:3]:
        render_card(r, "📌")
else:
    st.warning("❌ لم يتم العثور على نتائج.. وش رايك تسأل الذكي 👇")

    if st.button("🤖 الذكي"):
        model = load_model()
        descriptions = df[DESC_COL].fillna("").astype(str).tolist()
        embeddings = compute_embeddings(descriptions)

        query_embedding = model.encode(query, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_scores, top_indices = torch.topk(cosine_scores, k=min(3, len(df)))

        st.subheader("🔎:")
        for score, idx in zip(top_scores, top_indices):
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

if st.button("🔒 تسجيل خروج"):
    st.session_state.authenticated = False
    st.rerun()
