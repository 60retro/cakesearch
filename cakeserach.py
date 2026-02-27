import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

st.set_page_config(page_title="ค้นหาแบบเค้กพนักงาน Nami", layout="centered")

# ==========================================
# 1. เชื่อมต่อระบบ
# ==========================================
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("cakesearch")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
vision_model = genai.GenerativeModel('gemini-flash-lite-latest')

# ==========================================
# 2. โหลด AI Model แปลงข้อความ
# ==========================================
@st.cache_resource
def load_text_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

text_model = load_text_model()

# ==========================================
# 3. หน้าจอการค้นหา
# ==========================================
st.title("🎂 ระบบค้นหาแบบเค้ก Nami (AI โครงสร้างภาพ)")
st.write("ระบบจะประเมินรูปทรงและของตกแต่ง เพื่อหาแบบเค้กในร้านที่โครงสร้างเหมือนกันที่สุด (ไม่จำกัดสี)")

category_options = ["ค้นหาทั้งหมด (ไม่แยกหมวด)", "เค้ก 1 ชั้น", "เค้ก 2 ชั้น", "เค้ก 3 มิติ", "คัพเค้ก", "งานฟองดอง", "เค้กซ่อนเงิน", "เค้กดึงเงิน"]
selected_cat = st.selectbox("📌 เลือกหมวดหมู่ที่ต้องการค้นหา (ช่วยให้แม่นขึ้น 100%):", category_options)

search_file = st.file_uploader("อัพโหลดรูปภาพ Reference จากลูกค้า", type=["jpg", "jpeg", "png"])

if search_file:
    img_search = Image.open(search_file).convert("RGB")
    st.image(img_search, caption="รูปลูกค้า", width=250)

    if st.button("🔍 วิเคราะห์และค้นหา", type="primary", use_container_width=True):
        with st.spinner("AI กำลังถอดรหัสโครงสร้างเค้ก..."):
            try:
                # 1. ให้ Gemini ถอดรหัสภาพ
                prompt = """
                จงวิเคราะห์โครงสร้างของเค้กในรูปภาพนี้อย่างละเอียด ห้ามระบุสีเด็ดขาด 
                ให้ระบุเป็นคำสั้นๆ คั่นด้วยลูกน้ำ (,) โดยเน้นที่: จำนวนชั้น, รูปทรง, สไตล์การตกแต่งพื้นผิว, และของตกแต่ง
                """
                response = vision_model.generate_content([prompt, img_search])
                extracted_features = response.text.strip().replace('\n', ' ')
                
                st.success(f"**🧠 AI อ่านโครงสร้างภาพได้ว่า:** {extracted_features}")
                
                # 2. แปลงข้อความเป็น Vector แล้วค้นหา
                embedding = text_model.encode(extracted_features).tolist()
                
                query_params = {
                    "vector": embedding,
                    "top_k": 3,
                    "include_metadata": True
                }
                
                if selected_cat != "ค้นหาทั้งหมด (ไม่แยกหมวด)":
                    query_params["filter"] = {"category": selected_cat}

                response_pinecone = index.query(**query_params)

                # 3. แสดงผลลัพธ์
                if response_pinecone['matches']:
                    st.subheader(f"ผลการค้นหา:")
                    for match in response_pinecone['matches']:
                        meta = match['metadata']
                        
                        with st.container():
                            cols = st.columns([3, 2]) 
                            with cols[0]:
                                if meta.get('image_url'):
                                    st.image(meta['image_url'], use_container_width=True)
                            with cols[1]:
                                st.write(f"**ชื่อไฟล์:** {meta.get('filename', match['id'])}")
                                st.write(f"**หมวดหมู่:** {meta.get('category', 'ไม่ได้ระบุ')}")
                                st.caption(f"**คุณสมบัติภาพนี้:** {meta.get('description', '')}")
                                st.caption(f"ความตรงกันของโครงสร้าง: {match['score']:.2%}")
                            st.divider()
                else:
                    st.warning("ไม่พบแบบเค้กที่โครงสร้างตรงกันในหมวดนี้ครับ")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")

