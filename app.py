import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai
import json

# ดึง API Key
gemini_api_key = st.secrets["gemini_api_key"]
genai.configure(api_key=gemini_api_key)

# รายละเอียดฐานข้อมูล
db_name = 'test_database.db'
data_table = 'transactions'
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

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def query_to_dataframe(sql_query, database_name):
    """รัน SQL และคืนค่าเป็น DataFrame"""
    try:
        connection = sqlite3.connect(database_name)
        result_df = pd.read_sql_query(sql_query, connection)
        connection.close()
        return result_df
    except Exception as e:
        return f"Database Error: {e}"


def generate_gemini_answer(prompt, is_json=False):
    """เรียก Gemini API ด้วย google-generativeai SDK"""
    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite',
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json" if is_json else "text/plain"
            )
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# =============================================================
# PROMPT TEMPLATES
# =============================================================

script_prompt = """
### Goal
สร้าง SQLite script ที่สั้นและถูกต้องที่สุด เพื่อตอบคำถามจากข้อมูลที่มี โดยส่งออกเป็น JSON เท่านั้น

### Context
คุณคือ SQLite Master ที่ทำงานในระบบอัตโนมัติ (Strict JSON API) ห้ามตอบเป็นคำพูด ให้ตอบเฉพาะโค้ดที่ใช้งานได้จริง

### Input
- คำถามที่ผู้ใช้ต้องการคำตอบ: <Question> {question} </Question>
- ชื่อ Table ที่ต้องใช้ข้อมูล: <Table_Name> {table_name} </Table_Name>
- โครงสร้างของคอลัมน์: <Schema>
{data_dict}
</Schema>

### Process
1. วิเคราะห์ Query จาก <Question> และ <Schema>
2. หากมีเงื่อนไขวันที่ ให้ใช้ฟังก์ชัน `date()` หรือ `strftime()` ของ SQLite จัดการเสมอ
3. เขียน SQL ให้กระชับและมุ่งเน้นเฉพาะคำตอบที่ต้องการ

### Output
ตอบกลับเป็น JSON object รูปแบบเดียวเท่านั้น:
{{"script": "SELECT ... FROM ..."}}

(ห้ามมีข้อความอื่นประกอบ หรือ Markdown นอกเหนือจาก JSON)
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
5. Detecting Language: ตรวจสอบภาษาที่ใช้ใน <Question> และใช้ภาษานั้นในการตอบเสมอ (เช่น ถ้าถามอังกฤษให้ตอบอังกฤษ ถ้าถามไทยให้ตอบไทย)

### Output
ตอบเป็นข้อความสั้นๆ โดยใช้ภาษาเดียวกับ <Question> (Strictly match the requester's language) โดยมีโครงสร้างดังนี้:
1. คำเกริ่นนำ: ใช้ประโยคสั้นๆ เข้าประเด็นทันที (เช่น "From the data...", "Based on the information...")
2. เนื้อหา: ระบุผลการวิเคราะห์พร้อมตัวเลขที่ใส่คอมม่าและมีหน่วยลงท้ายเสมอ
"""

# =============================================================
# CORE LOGIC
# =============================================================

def generate_summary_answer(user_question):
    # 1. สร้าง SQL Prompt
    script_prompt_input = script_prompt.format(
        question=user_question,
        table_name=data_table,
        data_dict=data_dict_text
    )

    sql_json_text = generate_gemini_answer(script_prompt_input, is_json=True)

    try:
        # ป้องกัน Gemini ส่ง markdown code block กลับมา
        cleaned = sql_json_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        sql_script = json.loads(cleaned)['script']
    except Exception:
        return f"ขออภัย ไม่สามารถสร้างคำสั่ง SQL ได้\n\nGemini ตอบว่า:\n{sql_json_text}"

    # 2. Query ข้อมูล
    df_result = query_to_dataframe(sql_script, db_name)
    if isinstance(df_result, str):
        return df_result

    # 3. สรุปคำตอบ
    answer_prompt_input = answer_prompt.format(
        question=user_question,
        raw_data=df_result.to_string()
    )

    return generate_gemini_answer(answer_prompt_input, is_json=False)


# =============================================================
# USER INTERFACE
# =============================================================

# ตรวจสอบและสร้าง Chat History ใน Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title('Gemini Chat with Database')

# แสดงประวัติการสนทนา
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# รับ Input
if prompt := st.chat_input("พิมพ์คำถามที่นี่..."):
    # เก็บและแสดงข้อความ User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ประมวลผลและแสดงข้อความ Assistant
    with st.chat_message("assistant"):
        with st.spinner('กำลังหาคำตอบ...'):
            response = generate_summary_answer(prompt)
        st.markdown(response)

    # เก็บคำตอบลง Session
    st.session_state.messages.append({"role": "assistant", "content": response})
