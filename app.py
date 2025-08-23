import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# تحميل البيانات وكلمة المرور من Google Sheet
@st.cache_resource
def load_data_and_password():
    # قراءة بيانات Google credentials من secrets.toml
    creds_dict = dict(st.secrets["GOOGLE_CREDENTIALS"])
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )

    # الاتصال بجوجل شيت
    client = gspread.authorize(creds)
    sheet_id = st.secrets["SHEET"]["id"]  # موجود في secrets.toml
    sheet = client.open_by_key(sheet_id).sheet1

    # قراءة البيانات كلها
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    # جلب كلمة المرور من خلية محددة (E1 مثلاً)
    try:
        PASSWORD = sheet.acell("E1").value
    except Exception:
        PASSWORD = "1234"  # باسورد افتراضي لو ما حصل الخلية

    return df, PASSWORD


# تشغيل التطبيق
df, PASSWORD = load_data_and_password()

st.title("📊 Disaster App")

# إدخال كلمة المرور
user_pass = st.text_input("أدخل كلمة المرور:", type="password")

if user_pass == PASSWORD:
    st.success("تم تسجيل الدخول ✅")
    st.dataframe(df)
else:
    st.warning("أدخل كلمة المرور للمتابعة")
