import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import torch

# إعداد الصفحة
st.set_page_config(page_title="نظام إدارة الأزمات", layout="wide")

# --- إعدادات عامة ---
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# --- تحميل الموديل (مرة واحدة) ---
@st.cache_resource
def load_model():
    # لو تحب تحدد الجهاز CPU صراحةً:
    # torch.set_default_device("cpu")
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# --- الاتصال بالشيت وجلب البيانات (يُعاد كل 60 ثانية) ---
@st.cache_data(ttl=60)
def load_data_and_password():
    # نحصل على الدكت من st.secrets مباشرة (TOML → dict)
    creds_dict = dict(st.secrets["GOOGLE_CREDENTIALS"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)

    # نقرأ الشيت بالـ key
    sheet = client.open_by_key(st.secrets["SHEET"]["id"])
    ws = sheet.sheet1

    # البيانات
    data = ws.get_all_records()
    df = pd.DataFrame(data)

    # خلية كلمة المرور (إما من secrets أو من الشيت)
    if "password_cell" in st.secrets["SHEET"]:
        cell = st.secrets["SHEET"]["password_cell"]  # مثل "E1"
        row = int(''.join(filter(str.isdigit, cell)))
        col_letters = ''.join(filter(str.isalpha, cell)).upper()
        # تحويل حرف العمود إلى رقم بسيط (A=1, B=2, ...)
        col = sum((ord(c) - 64) * (26 ** i) for i, c in enumerate(reversed(col_letters)))
        password_value = ws.cell(row, col).value
    else:
        # التوافق مع طريقتك القديمة (العمود الخامس بالصف الأول)
        password_value = ws.cell(1, 5).value

    return df, password_value

# --- حساب الـ Embeddings (يُعاد فقط عند تغيّر البيانات) ---
@st.cache_data
def compute_embeddings(descriptions: list[str], model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    model = load_model()
    embs = model.encode(descriptions, convert_to_tensor=True)
    return embs

# --- واجهة المستخدم ---
st.title("🛠️ نظام إدارة الأزمات")

# جلب البيانات + كلمة المرور
df, PASSWORD = load_data_and_password()

# التحقق من وجود عمود الوصف
desc_col = 'وصف الحالة أو الحدث'
if desc_col not in df.columns:
    st.error(f"لم يتم العثور على العمود '{desc_col}' داخل Google Sheet. تأكد من اسم العمود بالضبط.")
    st.stop()

# إدخال كلمة المرور
user_password = st.text_input("أدخل كلمة المرور للوصول إلى البيانات:", type="password")

if user_password == PASSWORD and PASSWORD:
    st.success("تم التحقق من كلمة المرور ✅")
    st.subheader("📋 البيانات الحالية")
    st.dataframe(df, use_container_width=True)

    # إدخال وصف الحالة
    query = st.text_area("أدخل وصف الحالة أو الحدث:", max_chars=300)

    if query:
        descriptions = df[desc_col].fillna("").astype(str).tolist()
        embeddings = compute_embeddings(descriptions)

        query_embedding = load_model().encode([query], convert_to_tensor=True)
        scores = cosine_similarity(query_embedding, embeddings)[0]
        top_idx = int(torch.argmax(scores).item())

        st.markdown("### 🎯 أقرب حالة مشابهة:")
        st.write(df.iloc[top_idx])
else:
    st.warning("كلمة المرور غير صحيحة أو لم تُدخل بعد.")
