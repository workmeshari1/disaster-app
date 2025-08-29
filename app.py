import streamlit as st
import pandas as pd
import difflib

# إعداد الصفحة
st.set_page_config(page_title="البحث الذكي", page_icon="🔍", layout="centered")

# تحميل البيانات من Google Sheets
SHEET_URL = "ضع_رابط_Google_Sheet_الخاص_بك"
df = pd.read_csv(SHEET_URL)

# أعمدة البيانات
DESC_COL = "الوصف"
SYN_COL = "مرادفات للوصف"
ACTION_COL = "الإجراء"

# إدخال البحث
q = st.text_input("🔍 اكتب كلمة أو رقم للبحث:")

if q:
    # --------- 🔢 معالجة الأرقام ---------
    try:
        number = int(q)
        matched_action = None

        for _, row in df.iterrows():
            synonyms = str(row.get(SYN_COL, "")).replace(" ", "")

            # لو المرادفات مفصولة بفواصل
            for syn in synonyms.split(","):
                if not syn:
                    continue

                if "-" in syn:  # مكتوبة كمدى
                    parts = syn.split("-")
                    try:
                        min_val = int(parts[0])
                        max_val = 999999999 if parts[1] in ["∞", "inf"] else int(parts[1])
                    except:
                        continue

                    if min_val <= number <= max_val:
                        matched_action = row[ACTION_COL]
                        break
                else:  # قيمة مفردة
                    try:
                        if number == int(syn):
                            matched_action = row[ACTION_COL]
                            break
                    except:
                        continue

            if matched_action:
                break

        if matched_action:
            st.success(f"📌 {matched_action}")
            st.stop()

    except ValueError:
        pass  # مو رقم، يكمل البحث بالكلمات

# --------- 🔤 البحث بالكلمات بعد معالجة الأرقام ---------
matches = []
for _, row in df.iterrows():
    synonyms = str(row.get(SYN_COL, "")).split(",")
    for syn in synonyms:
        syn = syn.strip()
        if not syn:
            continue
        ratio = difflib.SequenceMatcher(None, q.lower(), syn.lower()).ratio()
        if ratio > 0.7:
            matches.append((row[DESC_COL], row[ACTION_COL]))
            break

if matches:
    for desc, action in matches:
        st.write(f"**{desc}** ➝ {action}")
else:
    st.warning("⚠️ لم يتم العثور على نتائج، جرّب كلمات أخرى.")

