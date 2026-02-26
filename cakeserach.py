import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import easyocr
from PIL import Image
import io
import re

st.set_page_config(page_title="Nami Custom Products", layout="centered")

# ==========================================
# 1. การเชื่อมต่อฐานข้อมูล (จาก Secrets)
# ==========================================
# เชื่อมต่อ Firebase
if not firebase_admin._apps:
    # ดึงข้อมูลจาก Streamlit Secrets
    firebase_creds = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred, {
        'storageBucket': st.secrets["FIREBASE_BUCKET"]
    })

# เชื่อมต่อ Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("custom-products")

# ==========================================
# 2. โหลด AI Models
# ==========================================
@st.cache_resource
def load_models():
    clip = SentenceTransformer('clip-ViT-B-32')
    ocr = easyocr.Reader(['th', 'en'], gpu=False)
    return clip, ocr

clip_model, ocr_reader = load_models()

def extract_price(text_list):
    full_text = " ".join(text_list)
    numbers = re.findall(r'\d+', full_text)
    return numbers[-1] if numbers else "ไม่ระบุราคา"

# ==========================================
# 3. หน้าจอการทำงาน (UI)
# ==========================================
st.title("🛍️ Nami Visual Search")

# สร้าง 2 แท็บ: สำหรับค้นหา และ สำหรับแอดมิน
tab_search, tab_admin = st.tabs(["🔍 ค้นหาสินค้า", "⚙️ จัดการสินค้า (Admin)"])

# -----------------------------------
# หน้าค้นหาสินค้า (ใช้งานได้ทุกคน)
# -----------------------------------
with tab_search:
    st.write("อัพโหลดรูปภาพเพื่อหาสินค้าที่ใกล้เคียงที่สุด")
    search_file = st.file_uploader("เลือกรูปภาพเพื่อค้นหา", type=["jpg", "jpeg", "png"], key="search")

    if search_file:
        img_search = Image.open(search_file).convert("RGB")
        st.image(img_search, caption="รูปที่ต้องการหา", width=250)

        if st.button("ค้นหาเลย", type="primary", use_container_width=True):
            with st.spinner("กำลังค้นหาข้อมูล..."):
                embedding = clip_model.encode(img_search).tolist()
                response = index.query(vector=embedding, top_k=3, include_metadata=True)

                if response['matches']:
                    st.subheader("สินค้าที่พบ:")
                    for match in response['matches']:
                        meta = match['metadata']
                        with st.container():
                            cols = st.columns([1, 2])
                            with cols[0]:
                                if meta.get('image_url'):
                                    st.image(meta['image_url'], use_container_width=True)
                            with cols[1]:
                                st.write(f"**รหัส/ชื่อ:** {match['id']}")
                                st.write(f"**ราคา:** {meta.get('price', 'ไม่ระบุ')} บาท")
                                st.caption(f"ความแม่นยำ: {match['score']:.2f}")
                            st.divider()
                else:
                    st.warning("ไม่พบสินค้าที่ใกล้เคียงครับ")

# -----------------------------------
# หน้าแอดมิน (ต้องใส่รหัสผ่าน)
# -----------------------------------
with tab_admin:
    st.write("อัพโหลดสินค้าใหม่เข้าระบบ")
    password = st.text_input("รหัสผ่าน Admin", type="password")
    
    if password == st.secrets["ADMIN_PASSWORD"]:
        st.success("เข้าสู่ระบบแอดมินสำเร็จ")
        
        # ให้ตั้งชื่อไฟล์ (ID) เพื่อไม่ให้ซ้ำกัน
        product_id = st.text_input("รหัสสินค้า / ชื่อรูปภาพ (ภาษาอังกฤษหรือตัวเลข)")
        upload_file = st.file_uploader("เลือกรูปภาพสินค้าเพื่ออัพโหลด", type=["jpg", "jpeg", "png"], key="upload")
        
        if upload_file and product_id:
            img_upload = Image.open(upload_file).convert("RGB")
            st.image(img_upload, width=250)
            
            if st.button("อัพโหลดขึ้นระบบ"):
                with st.spinner("กำลังประมวลผล (อ่านราคา -> อัพขึ้นคลาวด์ -> บันทึกข้อมูล)..."):
                    try:
                        # 1. ให้ AI อ่านราคา
                        img_byte_arr = io.BytesIO()
                        img_upload.save(img_byte_arr, format='JPEG')
                        ocr_result = ocr_reader.readtext(img_byte_arr.getvalue(), detail=0)
                        price = extract_price(ocr_result)
                        st.info(f"AI อ่านราคาได้: {price}")

                        # 2. อัพโหลดรูปขึ้น Firebase
                        bucket = storage.bucket()
                        blob = bucket.blob(f"products/{product_id}.jpg")
                        blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/jpeg')
                        blob.make_public()
                        image_url = blob.public_url

                        # 3. แปลงภาพเป็น Vector และบันทึกลง Pinecone
                        embedding = clip_model.encode(img_upload).tolist()
                        index.upsert(
                            vectors=[{
                                "id": product_id,
                                "values": embedding,
                                "metadata": {"price": price, "image_url": image_url}
                            }]
                        )
                        st.success(f"✅ บันทึกสินค้า {product_id} สำเร็จเรียบร้อยแล้ว!")
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {e}")
    elif password:

        st.error("รหัสผ่านไม่ถูกต้อง")
