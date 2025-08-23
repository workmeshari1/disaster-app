import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer
import json
from torch.nn.functional import cosine_similarity

# إعداد الصفحة
st.set_page_config(page_title="نظام إدارة الأزمات", layout="wide")

# تحميل البيانات والموديل
@st.cache_data(ttl=60)  # نخلي الكاش دقيقة وحدة عشان التحديث يكون أسرع
def load_data():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    # نقرأ من الـ secrets كـ dict مباشر
    creds_dict = dict(st.secrets["GOOGLE_CREDENTIALS"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets["SHEET"]["id"])
    data_sheet = sheet.sheet1

    data = data_sheet.get_all_records()
    df = pd.DataFrame(data)

    # كلمة المرور (مخزنة في العمود الخامس بالصف الأول)
    password_cell = data_sheet.cell(1, 5).value

    # تحميل الموديل
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    descriptions = df['وصف الحالة أو الحدث'].fillna("").astype(str).tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=True)

    return df, model, embeddings, password_cell

# واجهة التطبيق
st.title("🛠️ نظام إدارة الأزمات")

# إدخال كلمة المرور
user_password = st.text_input("أدخل كلمة المرور للوصول إلى البيانات:", type="password")

# تحميل البيانات
df, model, embeddings, PASSWORD = load_data()

# التحقق من كلمة المرور
if user_password == PASSWORD:
    st.success("تم التحقق من كلمة المرور ✅")
    st.subheader("📋 البيانات الحالية")
    st.dataframe(df, use_container_width=True)

    # إدخال وصف الحالة
    query = st.text_area("أدخل وصف الحالة أو الحدث:", max_chars=300)

    if query:
        query_embedding = model.encode([query], convert_to_tensor=True)
        scores = cosine_similarity(query_embedding, embeddings)[0]
        top_idx = scores.argmax().item()

        st.markdown("### 🎯 أقرب حالة مشابهة:")
        st.write(df.iloc[top_idx])
else:
    st.warning("كلمة المرور غير صحيحة أو لم تُدخل بعد.")
