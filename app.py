import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

# --- الخلفية مع إزاحة للأسفل + إخفاء الشعار والأيقونات + تصغير العناوين ---
page_style = f"""
<style>
.stApp {{
    background-image: url("https://github.com/workmeshari1/disaster-app/blob/bb4be38238ac06288848fa086e098f56b21e92b4/assets.png?raw=true");
    background-size: cover;
    background-position: top center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    padding-top: 80px;  /* نزول الخلفية تحت الهيدر */
}}

/* للشاشات الصغيرة (جوال) */
@media only screen and (max-width: 768px) {{
    .stApp {{
        background-size: contain;
        background-position: top center;
        padding-top: 60px;
    }}
}}

/* إخفاء شعار streamlit والفوتر */
#MainMenu, header, footer {{
    visibility: hidden;
}}

/* إخفاء الأيقونات الصغيرة أسفل البحث */
.st-emotion-cache-12fmjuu, 
[data-testid="stDecoration"], 
.stDeployButton {{
    display: none !important;
}}

/* تصغير العناوين */
h1 {{
    font-size: 26px !important;
    color: #ffffff;
    text-align: center;
    margin-top: -60px; /* يرفع العنوان فوق الخلفية */
}}
h2 {{
    font-size: 20px !important;
    color: #ffffff;
}}
h3 {{
    font-size: 18px !important;
    color: #ffffff;
}}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

st.set_page_config(page_title="⚡ إدارة الكوارث والأزمات", layout="centered", initial_sidebar_state="collapsed")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# --- تحميل الموديل مرة واحدة ---
@st.cache_resource
def load_model():
    return SentenceTransformer("Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka")

# --- قراءة البيانات + كلمة المرور من الشيت (كل 10 دق) ---
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

# --- حساب إمبادنج للوصف ---
@st.cache_data
def compute_embeddings(descriptions: list[str]):
    model = load_model()
    return model.encode(descriptions, convert_to_tensor=True)

# --- دالة للتحقق من الأرقام ضمن نطاق أو قيمة مفردة ---
def is_number_in_range(number, synonym):
    """
    تحقق مما إذا كان الرقم يقع ضمن نطاق أو يساوي قيمة مفردة.
    
    Args:
        number (int): الرقم المدخل للمقارنة.
        synonym (str): النطاق (مثل "10-20") أو القيمة المفردة (مثل "15").
    
    Returns:
        bool: True إذا كان الرقم يتطابق مع النطاق أو القيمة، False عكس ذلك.
    """
    try:
        if "-" in synonym:
            parts = synonym.split("-")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                return False
            min_val = int(parts[0])
            max_val = float('inf') if parts[1] in ["∞", "inf"] else int(parts[1])
            return min_val <= number <= max_val
        else:
            return number == int(synonym)
    except ValueError as e:
        print(f"خطأ في معالجة القيمة أو النطاق '{synonym}': {e}")
        return False

# --- دالة لمعالجة الرقم المدخل ---
def process_number_input(q, df, syn_col, action_col):
    """
    معالجة الرقم المدخل والبحث عن إجراء مطابق في DataFrame.
    
    Args:
        q (str): المدخل (يفترض أنه رقم).
        df (pd.DataFrame): DataFrame يحتوي على أعمدة المرادفات والإجراءات.
        syn_col (str): اسم عمود المرادفات.
        action_col (str): اسم عمود الإجراءات.
    
    Returns:
        bool: True إذا تم العثور على تطابق وتم عرضه، False عكس ذلك.
    """
    try:
        number = int(q)
        matched_action = None

        for _, row in df.iterrows():
            synonyms = str(row.get(syn_col, "")).strip()
            if not synonyms:
                continue
                
            for syn in synonyms.split(","):
                syn = syn.strip()
                if not syn:
                    continue
                if is_number_in_range(number, syn):
                    matched_action = row[action_col]
                    break
            if matched_action:
                break

        # التحقق من النطاقات المحددة (2000-4999 و5000-∞)
        if matched_action:
            st.success(f"📌 {matched_action}")
            return True
        else:
            st.warning("لم يتم العثور على تطابق للرقم المدخل.")
            return False

    except ValueError:
        return False  # مو رقم، ينتقل للبحث النصي

    except ValueError:
        return False  # مو رقم، ينتقل للبحث النصي

# ============== واجهة ==============
st.title("⚡ دائرة إدارة الكوارث والأزمات الصناعية")

# جرب تحميل البيانات
try:
    df, PASSWORD = load_data_and_password()
except Exception as e:
    st.error(f"❌ فشل الاتصال بقاعدة البيانات: {str(e)}")
    st.info("تأكد من إعداد متغيرات البيئة أو أسرار Streamlit بشكل صحيح.")
    st.stop()

# التحقق من الأعمدة المطلوبة
DESC_COL = "وصف الحالة أو الحدث"
ACTION_COL = "الإجراء"
SYN_COL = "مرادفات للوصف"

if df.empty:
    st.error("❌ لا توجد بيانات في الجدول. تأكد من وجود بيانات في Google Sheet.")
    st.stop()

for col in [DESC_COL, ACTION_COL]:
    if col not in df.columns:
        st.error(f"عمود مفقود في Google Sheet: '{col}'. تأكد من اسم العمود حرفيًا.")
        st.info(f"الأعمدة المتاحة: {list(df.columns)}")
        st.stop()

if SYN_COL not in df.columns:
    df[SYN_COL] = ""  # نضيفه فارغ إذا ناقص

# تسجيل الدخول
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("الرقم السري", type="password")
    if st.button("دخول"):
        if password == str(PASSWORD):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ الرقم السري غير صحيح")
    st.stop()

# بعد التحقق
query = st.text_input("ابحث هنا:", placeholder="اكتب وصف الحالة…")
if not query:
    st.stop()

q = query.strip().lower()

# --------- 🔢 معالجة الأرقام مع دعم النطاقات والقيم المتعددة ---------
if process_number_input(q, df, SYN_COL, ACTION_COL):
    st.stop()

# --------- 📝 البحث النصي ---------
words = [w for w in q.split() if w]
literal_results = []
synonym_results = []

# 1) الحرفي من الوصف
for _, row in df.iterrows():
    text = str(row[DESC_COL]).lower()
    if all(w in text for w in words):
        literal_results.append(row)

# 2) الحرفي من المرادفات
if not literal_results:
    for _, row in df.iterrows():
        syn_text = str(row.get(SYN_COL, "")).lower()
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
    st.subheader("🔍 النتائج المطابقة:")
    for r in literal_results[:3]:
        render_card(r, "🔍")
elif synonym_results:
    st.subheader("📌 يمكن قصدك:")
    for r in synonym_results[:3]:
        render_card(r, "📌")
else:
    st.warning("❌ لم يتم العثور على نتائج.. وش رايك تسأل الذكي 👇")
    if st.button("🤖 اسأل الذكي"):
        try:
            with st.spinner("جاري البحث الذكي..."):
                model = load_model()
                descriptions = df[DESC_COL].fillna("").astype(str).tolist()
                
                if not descriptions or all(not desc.strip() for desc in descriptions):
                    st.error("❌ لا توجد أوصاف صالحة في البيانات.")
                    st.stop()
                
                embeddings = compute_embeddings(descriptions)
                query_embedding = model.encode(query, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
                top_scores, top_indices = torch.topk(cosine_scores, k=min(3, len(df)))

                st.subheader("🧐 يمكن قصدك:")
                found_results = False
                for score, idx in zip(top_scores, top_indices):
                    if float(score) > 0.3:
                        found_results = True
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
                
                if not found_results:
                    st.info("🤖 لم يتم العثور على نتائج مشابهة كافية. حاول إعادة صياغة سؤالك.")
                    
        except Exception as e:
            st.error(f"❌ خطأ في البحث الذكي: {str(e)}")

# شريط جانبي للمعلومات الإضافية
with st.sidebar:
    st.markdown("### معلومات النظام")
    st.info(f"📊 عدد الحالات المسجلة: {len(df)}")
    st.info("🔄 تحديث البيانات: كل 10 دقائق")
    
    if st.button("🔒 تسجيل خروج"):
        st.session_state.authenticated = False
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; direction: rtl;'>
    آلية إدارة الكوارث والأزمات الذكية
    </div>
    """,
    unsafe_allow_html=True
)

