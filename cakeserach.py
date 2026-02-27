import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
import threading
import os
import io
import time
from PIL import Image
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, storage
from pinecone import Pinecone
import hashlib
from google import genai

# ==========================================
# 1. ตั้งค่า API Keys 
# ==========================================
PINECONE_API_KEY = "pcsk_5RgbxU_83K2o4mCWNqq1JvofmMY8ovLHrzKSnVanaWaBzFZNWob2Raks3L8f9iSNzbW9Ri"
PINECONE_INDEX_NAME = "cakesearch"
FIREBASE_BUCKET = "cakesearch-541d5.firebasestorage.app"
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin" 

# [จุดที่ต้องแก้]: ใส่คีย์ Gemini API ของคุณตรงนี้ครับ
GEMINI_API_KEY = "ใส่_API_KEY_ของ_GEMINI_ของคุณตรงนี้"
client = genai.Client(api_key=GEMINI_API_KEY)

class NamiUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ระบบอัพโหลด Nami (AI วิเคราะห์โครงสร้างเค้ก)")
        self.root.geometry("650x600")
        self.root.configure(padx=20, pady=20)

        self.selected_folder = tk.StringVar()
        self.category_name = tk.StringVar() 

        tk.Label(root, text="อัพโหลดรูปและให้ AI สกัดโครงสร้าง", font=("Helvetica", 16, "bold")).pack(pady=(0, 15))

        frame_folder = tk.Frame(root)
        frame_folder.pack(fill=tk.X, pady=5)
        tk.Label(frame_folder, text="โฟลเดอร์ไฟล์:").pack(side=tk.LEFT)
        tk.Entry(frame_folder, textvariable=self.selected_folder, state='readonly', width=45).pack(side=tk.LEFT, padx=10)
        tk.Button(frame_folder, text="เลือก...", command=self.browse_folder).pack(side=tk.LEFT)

        frame_cat = tk.Frame(root)
        frame_cat.pack(fill=tk.X, pady=10)
        tk.Label(frame_cat, text="หมวดหมู่สินค้า (เช่น เค้ก 2 ชั้น):", fg="blue", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        tk.Entry(frame_cat, textvariable=self.category_name, width=30).pack(side=tk.LEFT, padx=10)

        self.btn_start = tk.Button(root, text="🚀 เริ่มให้ AI วิเคราะห์และอัพโหลด", font=("Helvetica", 12), bg="#4CAF50", fg="white", command=self.start_processing)
        self.btn_start.pack(pady=15, fill=tk.X)

        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 10))

        tk.Label(root, text="สถานะการทำงาน:").pack(anchor=tk.W)
        self.log_area = scrolledtext.ScrolledText(root, height=15, state='disabled', bg="#f0f0f0")
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.selected_folder.set(folder_path)

    def log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        self.root.update_idletasks()

    def start_processing(self):
        folder = self.selected_folder.get()
        cat_name = self.category_name.get().strip()

        if not folder or not cat_name:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเลือกโฟลเดอร์ และ พิมพ์ชื่อหมวดหมู่ก่อนครับ!")
            return

        self.btn_start.config(state='disabled')
        self.progress['value'] = 0
        self.log(f"เริ่มทำงาน... หมวดหมู่: [{cat_name}]")

        threading.Thread(target=self.process_files, args=(folder, cat_name), daemon=True).start()

    def process_files(self, folder, cat_name):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate("firebase_key.json")
                firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_BUCKET})
            
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)

            self.log("กำลังโหลดโมเดลแปลงข้อความ (ภาษาไทย)...")
            text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
            if not files:
                self.log(f"ไม่พบไฟล์รูปภาพในโฟลเดอร์เลยครับ")
                self.btn_start.config(state='normal')
                return

            self.log(f"พบ {len(files)} ไฟล์ เริ่มประมวลผล...\n" + "-"*40)
            self.progress['maximum'] = len(files)

            prompt = """
            จงวิเคราะห์โครงสร้างของเค้กในรูปภาพนี้อย่างละเอียดเพื่อใช้เป็นคีย์เวิร์ดค้นหา 
            ห้ามระบุสีเด็ดขาด ให้ระบุเป็นคำสั้นๆ คั่นด้วยลูกน้ำ (,) โดยเน้นที่:
            1. จำนวนชั้น (เช่น เค้ก 1 ชั้น, เค้ก 2 ชั้น)
            2. รูปทรง (เช่น ทรงกลม, ทรงสูง, ทรงหัวใจ)
            3. สไตล์การตกแต่งพื้นผิว (เช่น ปาดเรียบ, ปาดลายคลื่น, งานฟองดอง)
            4. ของตกแต่ง (เช่น ดอกไม้, ตุ๊กตา, มงกุฎ, ป้ายอักษร, ผลไม้, มาการอง)
            """

            for i, filename in enumerate(files):
                filepath = os.path.join(folder, filename)
                images_to_process = []
                self.log(f"จัดการ: {filename}")
                
                if filename.lower().endswith('.pdf'):
                    try:
                        pages = convert_from_path(filepath, poppler_path=POPPLER_PATH) 
                        for j, page in enumerate(pages):
                            images_to_process.append((f"{filename.replace('.pdf', '')}_p{j+1}.jpg", page))
                    except:
                        continue
                else:
                    try:
                        img = Image.open(filepath).convert("RGB")
                        images_to_process.append((filename, img))
                    except:
                        continue

                for img_name, img in images_to_process:
                    success = False
                    retries = 0
                    max_retries = 3

                    # ระบบ Retry วนซ้ำถ้าระบบแจ้งโควต้าเต็ม
                    while not success and retries < max_retries:
                        try:
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[prompt, img]
                            )
                            cake_description = response.text.strip().replace('\n', ' ')
                            self.log(f"   -> AI มองเห็น: {cake_description}")

                            embedding = text_model.encode(cake_description).tolist()

                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='JPEG')
                            bucket = storage.bucket()
                            safe_name = img_name.replace(' ', '_')
                            blob = bucket.blob(f"products/{safe_name}")
                            blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/jpeg')
                            blob.make_public()
                            
                            pinecone_id = hashlib.md5(img_name.encode('utf-8')).hexdigest()
                            index.upsert(
                                vectors=[{
                                    "id": pinecone_id,
                                    "values": embedding,
                                    "metadata": {
                                        "image_url": blob.public_url,
                                        "filename": img_name,
                                        "category": cat_name,
                                        "description": cake_description
                                    }
                                }]
                            )
                            self.log(f"   -> ✅ อัพโหลดสำเร็จ!")
                            success = True
                            
                            # เพิ่มเวลาพักเป็น 5 วินาทีให้ชัวร์ขึ้น
                            time.sleep(5) 
                            
                        except Exception as e:
                            error_msg = str(e)
                            # ถ้าเป็น Error โควต้า 429 ให้รอ 1 นาทีแล้วลุยต่อ
                            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                                self.log("   -> ⏳ โควต้ารายนาทีเต็ม! ระบบขอพักหายใจ 60 วินาที...")
                                time.sleep(60)
                                retries += 1
                            else:
                                self.log(f"   -> ❌ [Error]: {e}")
                                break # ถ้าเป็น Error อื่นๆ ให้ข้ามไปรูปต่อไป
                
                self.progress['value'] = i + 1
                self.log("-" * 40)

            self.log("\n🎉 ดำเนินการเสร็จสิ้นทั้งหมดแล้ว!")
            messagebox.showinfo("สำเร็จ", "อัพโหลดและวิเคราะห์ข้อมูลเสร็จเรียบร้อยแล้วครับ!")

        except Exception as e:
            self.log(f"\n❌ Error: {e}")
        finally:
            self.btn_start.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = NamiUploaderApp(root)
    root.mainloop()
