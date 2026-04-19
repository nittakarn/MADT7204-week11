import streamlit as st
import pandas as pd
import sqlite3
from google import genai
from google.genai import types
import json

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# ดึง API Key จาก Streamlit Secrets
# (ถ้าใช้ใน Colab ให้เปลี่ยนเป็น gemini_api_key = "YOUR_KEY")
try:
    gemini_api_key = st.secrets["gemini_api_key"]
except:
    gemini_api_key = "YOUR_GEMINI_API_KEY_HERE"

gmn_client = genai.Client(api_key=gemini_api_key)

db_name = 'test_database.db'
data_table = 'transactions'

# รายละเอียดคอลัมน์เพื่อให้ AI เข้าใจฐานข้อมูล
data_dict_text = """
- trx_date: วันที่ทำธุรกรรม
- trx_no: หมายเลขธุรกรรม
- member_code: รหัสสมาชิกของลูกค้า
- branch_code: รหัสสาขา
- branch_region: ภูมิภาคที่สาขาตั้งอยู่
- branch_province: จังหวัดที่สาขาตั้งอยู่
- product_code: รหัสสินค้า
- product_category: หมวดหมู่หลักของสินค้า
- product_group: กลุ่มของสินค้า
- product_type: ประเภทของสินค้า
- order_qty: จำนวนชิ้น/หน่วย ที่ลูกค้าสั่งซื้อ
- unit_price: ราคาขายของสินค้าต่อ 1 หน่วย
- cost: ต้นทุนของสินค้าต่อ 1 หน่วย
- item_discount: ส่วนลดเฉพาะรายการสินค้านั้นๆ
- customer_discount: ส่วนลดจากสิทธิของลูกค้า
- net_amount: ยอดขายสุทธิของรายการนั้น
- cost_amount: ต้นทุนรวมของรายการนั้น
"""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def query_to_dataframe(sql_query, database_name):
    """รันคำสั่ง SQL และคืนค่าเป็น DataFrame"""
    try:
        connection = sqlite3.connect(database_name)
        result_df = pd.read_sql_query(sql_query, connection)
        connection.close()
        return result_df
    except Exception as e:
        return f"Database Error: {e}"

def generate_gemini_answer(prompt, is_json=False):
    """เรียกใช้งาน Gemini API"""
    try:
        # เลือกโหมดการตอบกลับ (JSON หรือ Text)
        if is_json:
            config = types.GenerateContentConfig(response_mime_type="application/json")
        else:
            config = types.GenerateContentConfig(response_mime_type="text/plain")
            
        response = gmn_client.models.generate_content(
            model='gemini-2.0-flash', # หรือใช้ gemini-1.5-flash ตามที่รองรับ
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# ==========================================
# 3. PROMPT TEMPLATES
# ==========================================

script_prompt = """
### Goal
สร้าง SQLite script ที่สั้นและถูกต้องที่สุดเพื่อตอบคำถามจากข้อมูลที่มี โดยส่งออกเป็น JSON เท่านั้น

### Context
คุณคือ SQLite Master ที่ทำงานในระบบอัตโนมัติ (Strict JSON API) ห้ามตอบเป็นคำพูด ให้ตอบเฉพาะโค้ดที่ใช้งานได้จริง

### Input
- คำถามที่ผู้ใช้ต้องการคำตอบ: <Question> {question} </Question>
- ชื่อ Table ที่ต้องใช้ดึงข้อมูล: <Table_Name> {table_name} </Table_Name>
- คำอธิบายคอลัมน์: <Schema>
{data_dict}
</Schema>

### Process
1. วิเคราะห์ Query จาก <Question> และ <Schema>
2. หากมีคอลัมน์วันที่ ให้ใช้ฟังก์ชัน `date()` หรือ `strftime()` ของ SQLite จัดการเสมอ
3. เขียน SQL ให้กระชับและมุ่งเน้นเฉพาะคำตอบที่ต้องการ

### Output
ตอบกลับเป็น JSON object รูปแบบเดียวเท่านั้น:
{{"script": "SELECT ... FROM ..."}}
(ห้ามมีคำอธิบายประกอบ หรือ Markdown นอกเหนือจาก JSON)
"""

answer_prompt = """
### Goal
สรุปผลลัพธ์จากข้อมูลและตอบคำถามอย่างถูกต้อง แม่นยำ และเป็นธรรมชาติ

### Context
คุณคือ Data Analyst ที่ทำหน้าที่สรุปผลจาก DataFrame และตอบคำถามผู้ใช้แบบเจาะจง ห้ามตอบยาวเกินความจำเป็น และเน้นการวิเคราะห์เชิงตัวเลขที่ถูกต้อง

### Input
- คำถามที่ผู้ใช้ต้องการคำตอบ: <Question> {question} </Question>
- ข้อมูลจาก DataFrame: <Raw_Data>
{raw_data}
</Raw_Data>

### Process
1. วิเคราะห์ข้อมูลจาก <Raw_Data> ให้สอดคล้องกับ <Question>
2. คำนวณและสรุปข้อมูลเชิงสถิติที่สำคัญ
3. จัดรูปแบบตัวเลข: ใส่คอมม่า ( , ) คั่นหลักพัน และทศนิยมไม่เกิน 2 ตำแหน่ง
4. ระบุหน่วย ( เช่น บาท, คน, ครั้ง, % ) ต่อท้ายตัวเลขทุกครั้งตามบริบทของข้อมูล
5. ตรวจสอบภาษาที่ใช้ใน <Question> และใช้ภาษานั้นในการตอบกลับเสมอ (ถ้าถามไทยตอบไทย ถามอังกฤษตอบอังกฤษ)

### Output
ตอบเป็นข้อความสั้นๆ โดยใช้โครงสร้างดังนี้:
1. คำเกริ่นนำ: เช่น "จากข้อมูลพบว่า...", "Based on the data..."
2. เนื้อหา: ระบุผลการวิเคราะห์พร้อมตัวเลขและหน่วยที่ถูกต้อง
"""

# ==========================================
# 4. CORE LOGIC
# ==========================================

def generate_summary_answer(user_question):
    # ขั้นตอนที่ 1: ให้ AI สร้าง SQL Script
    prompt_for_sql = script_prompt.format(
        question=user_question,
        table_name=data_table,
        data_dict=data_dict_text
    )
    
    sql_json_text = generate_gemini_answer(prompt_for_sql, is_json=True)
    
    try:
        # ทำความสะอาดสตริงเผื่อ AI ใส่ Markdown มา
        clean_json = sql_json_text.replace("```json", "").replace("```", "").strip()
        sql_script = json.loads(clean_json)['script']
    except:
        return "ขออภัย ไม่สามารถสร้างคำสั่ง SQL สำหรับคำถามนี้ได้"

    # ขั้นตอนที่ 2: ดึงข้อมูลจากฐานข้อมูล
    df_result = query_to_dataframe(sql_script, db_name)
    
    # ถ้า Query พัง
    if isinstance(df_result, str):
        return f"เกิดข้อผิดพลาดในการดึงข้อมูล: {df_result}"
    
    # ถ้าไม่มีข้อมูลกลับมา
    if df_result.empty:
        return "ไม่พบข้อมูลที่ตรงกับเงื่อนไขที่คุณสอบถาม"

    # ขั้นตอนที่ 3: ให้ AI สรุปผลลัพธ์เป็นภาษามนุษย์
    prompt_for_answer = answer_prompt.format(
        question=user_question,
        raw_data=df_result.head(50).to_string() # จำกัดข้อมูลไว้ 50 แถวกัน Token เต็ม
    )
    
    final_answer = generate_gemini_answer(prompt_for_answer, is_json=False)
    return final_answer

# ==========================================
# 5. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Gemini SQL Chat", page_icon="📊")

# สร้างประวัติการแชทใน Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title('🚀 Gemini Chat with Database')
st.caption("ถามข้อมูลยอดขาย รายการธุรกรรม หรือสรุปผลจาก Database ได้ทันที")

# แสดงประวัติการสนทนา
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ส่วนรับคำถามจากผู้ใช้
if prompt := st.chat_input("ตัวอย่าง: ขอยอดขายรวมของเดือนมกราคม 2026"):
    # 1. แสดงข้อความผู้ใช้
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. ประมวลผลและแสดงข้อความจาก AI (Assistant)
    with st.chat_message("assistant"):
        with st.spinner('กำลังวิเคราะห์ฐานข้อมูลและหาคำตอบ...'):
            response = generate_summary_answer(prompt)
            st.markdown(response)

    # 3. บันทึกคำตอบลงประวัติ
    st.session_state.messages.append({"role": "assistant", "content": response})
