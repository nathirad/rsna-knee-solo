# RSNA Knee Abnormality Detection: งานส่วนของเรา

โค้ดและบันทึกการทดลองของเราในการแข่ง Kaggle [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) ช่วง 4–10 ตุลาคม 2026 (ทีม Misa-Amane, บัญชี Kaggle `natchapon600176`)

โจทย์: รับ MRI เข่าหนึ่งเคส (หลาย series) แล้วให้ความน่าจะเป็นของ 12 findings วัดด้วย macro AUC เทียบกับ label ของผู้เชี่ยวชาญ ส่งเป็น notebook ที่รันไม่เกิน 9 ชั่วโมงโดยไม่ต่ออินเทอร์เน็ต

## ผลบน public leaderboard

| วันที่ | สิ่งที่ส่ง | คะแนน |
|---|---|---|
| 6 ต.ค. | stack สาธารณะ + 2.5D reader 5 fold (เราเทรน fold 3, 4 เพิ่มเอง) | 0.944 |
| 9 ต.ค. | เหมือนบรรทัดบน แต่ Effusion/Synovitis ของ reader มาจากโมเดลเก่าของเรา | 0.944 |
| 9 ต.ค. | เหมือนบรรทัดแรก แต่หัวของ reader เทรนใหม่ด้วย label ที่ปรับแล้ว | 0.944 |
| 10 ต.ค. | โมเดล CoAtNet สาธารณะตัวเดียว ไม่แก้อะไร | 0.949 |
| 10 ต.ค. | โมเดลนั้น + reader ของเราที่น้ำหนัก 0.30 | 0.949 |

คะแนน 0.949 มาจากโมเดลสาธารณะของคนอื่น ไม่ใช่โมเดลที่เราเทรนเอง ส่วนที่เป็นงานของเราคือทุกอย่างที่อธิบายด้านล่าง ซึ่งจนถึงตอนนี้ยังไม่ได้ทำให้คะแนนบน leaderboard ขยับเกินฐานที่ใช้

## สิ่งที่เราทำ

### 1. โมเดลของเราเอง 5 fold
ConvNeXt-T ที่ 224 px บน cache 9 สไลซ์ต่อ series เทรนด้วย label จาก report โดยให้น้ำหนักน้อยกับช่องที่ report ไม่ได้พูดถึง
- out-of-fold 0.873, บนเคสที่ผู้เชี่ยวชาญ label 58 เคส (gold-58) ได้ 0.882
- ตรวจแล้วว่าเส้นทางตอนส่ง (อ่านจาก DICOM) ให้ pixel เหมือนกับ cache ตอนเทรนทุกเคสที่ตรวจ 300 เคส
- โค้ด: `kaggle/rsna-knee-solo/`, `kaggle/rsna-knee-train-solo/`

### 2. เทรน 2.5D reader บน Kaggle T4
นำสูตร 2.5D ConvNeXt reader ที่เผยแพร่สาธารณะมาเทรนเองบน GPU T4 ของ Kaggle ซึ่งต้องแก้หลายจุดให้รันได้ในโควต้า
- cache ปริมาตรทั้งชุดเป็น shard แล้วอ่านตรงจาก shard (แตกไฟล์ 110 GB ลงดิสก์ทำให้ session ตาย)
- fp16, micro-batch พร้อม gradient accumulation, ตรวจหน่วยความจำก่อนเทรน, วัดเวลาแล้วลดจำนวน epoch ถ้าไม่ทัน
- ตรวจว่า cache ของเราให้ผลเท่ากับเส้นทางของต้นฉบับ (ต่างกันไม่เกิน 0.00015)
- เทรน fold 3 และ 4 ได้ gold-58 0.899 และ 0.906 รวมกับ 3 fold สาธารณะเป็น 5 fold ได้ 0.913
- ลอง ConvNeXtV2-T (ไม่ดีกว่า ช้ากว่า 2 เท่า) และ ResNet-50 (แย่กว่ามาก)
- โค้ด: `kaggle/rsna-knee-gpjw/` (`train_gpjw.py`, `cache_cell.py`, `shards.py`, `port_check.py`, `build_*.py`)

### 3. วิธีเทรนเฉพาะ "หัว" ของโมเดลบน CPU
โมเดลแบ่งเป็นตัวดูภาพ (backbone) กับหัวที่ตัดสินผล เราบันทึกสิ่งที่ backbone เห็นของทุกเคสไว้ครั้งเดียว แล้วเทรนหัวใหม่ได้ในไม่กี่นาทีโดยไม่ใช้ GPU ทำให้เทียบนโยบาย label ได้เร็วมาก
- `feats_gpjw.py` ดึง feature, `research/head_lab.py` ทดลอง, `fit_heads.py` เทรนหัวแล้วประกอบกลับเป็น checkpoint ที่ใช้ส่งได้
- หัวที่เทรนใหม่ด้วย label เดิมได้ผลเท่ากับ checkpoint เดิม (0.909 เทียบกับ 0.908) จึงใช้เป็นเครื่องมือเทียบได้

### 4. ข้อค้นพบเรื่อง label
label ที่ได้จาก report กับ label ของผู้เชี่ยวชาญไม่ได้หมายถึงสิ่งเดียวกัน บน gold-58 สัดส่วนที่ผู้เชี่ยวชาญให้เป็นบวกขึ้นกับระดับความหนักแน่นของ report:

| finding | report ไม่ฟันธง | report ปกติ | report ชัดเจน |
|---|---|---|---|
| ACL | 0% | 20% | 100% |
| MCL | 0% | 70% | 100% |
| Effusion | 48% | 89% | 100% |
| Contusion | 22% | 46% | 100% |

แต่ label ที่ใช้เทรนกันทั่วไปให้ทุกระดับเป็นบวกราว 0.9 เราจึงสร้างเป้าหมายใหม่เป็น P(ผู้เชี่ยวชาญให้บวก) โดย fit จาก gold-58 (`research/label_calibration.py`, `make_labels_calibrated.py`)
- หัวของ reader ที่เทรนด้วยเป้าหมายใหม่: gold-58 ของ 5 fold 0.9130 → 0.9187, Effusion 0.877 → 0.924
- ตรวจโดย fit จาก gold ครึ่งหนึ่งแล้ววัดกับอีกครึ่ง ผลยังอยู่ (+0.0095 บน 2 fold)

### 5. การ validate
- gold-58, out-of-fold, paired bootstrap ทุกการเปรียบเทียบ
- เขียนเกณฑ์ตัดสินก่อนเห็นผล แล้วทำตามเกณฑ์ (เช่น รอบ fine-tune ที่ไม่ผ่านเกณฑ์ก็ไม่ส่ง)
- การเลือกที่ทำจาก gold-58 ตรวจซ้ำด้วยการเลือกจากครึ่งหนึ่งแล้ววัดอีกครึ่ง
- สคริปต์: `research/gpjw_*_check.py`, `research/gold_blend_check.py`, `research/per_finding_sources.py`

### 6. ต่อยอดจากโมเดลสาธารณะตัวเดียว
- ดึง feature ของโมเดลนั้นทั้ง 4,407 เคส (`kaggle/rsna-knee-raptor/`) แก้ปัญหาหน่วยความจำเต็มเมื่อรันกับชุดเทรนทั้งชุด
- เทรนหัวใหม่ให้มันด้วย label ของเรา: ไม่ดีกว่าหัวเดิม (0.9219 เทียบกับ 0.915 หรือต่ำกว่า)
- พบว่ามันอ่อนที่ Lateral OA (0.774 บน gold-58 ขณะที่ reader ของเราได้ 0.878) และเปลี่ยนหัวก็ไม่ดีขึ้น จึงทำ notebook ที่ยืมเฉพาะคอลัมน์นั้นจาก reader ของเรา (ส่งแล้ว รอคะแนน)

## สิ่งที่ลองแล้วไม่ได้ผล

- ผสม label กับคำทำนาย out-of-fold แบบ 50/50: ดีขึ้นกับ label จาก report แต่ไม่ดีขึ้นกับ label ผู้เชี่ยวชาญ
- fine-tune ทั้งโมเดล 4 epoch ด้วย label ใหม่: ขยับน้อยกว่าเทรนหัวใหม่จากศูนย์
- ใส่เคส gold ลงไปเทรนหัวด้วย: จำเคสที่เห็นได้ แต่ไม่ช่วยเคสที่ไม่เห็น
- mirror TTA กับ reader: ได้จริงแต่เล็กมาก (+0.002) และใช้เวลาเพิ่มเท่าตัว
- ผสม reader ของเราเข้ากับ stack หรือกับโมเดลสาธารณะตัวเดียว: gold-58 บอกว่าดีขึ้น แต่ leaderboard ไม่ขยับ ข้อสรุปคือ gold-58 ใช้ตัดสินการผสมโมเดลไม่ได้
- ปรับน้ำหนักการผสมราย finding: เมื่อเลือกจากครึ่งหนึ่งของ gold แล้ววัดอีกครึ่ง ไม่ดีกว่าน้ำหนักเดียว

รายละเอียดและตัวเลขทุกรอบอยู่ใน `runs/LOG.md`

## โครงสร้าง

```
kaggle/rsna-knee-gpjw/        เทรน reader บน T4, ดึง feature, เทรนหัว, ประกอบ notebook สำหรับส่ง
kaggle/rsna-knee-raptor/      ดึง feature และทดลองหัวของโมเดลสาธารณะตัวเดียว, notebook ผสมราย finding
kaggle/rsna-knee-solo/        โมเดลของเราเอง: ตัวโมเดลและเส้นทางตอนส่ง
kaggle/rsna-knee-train-solo/  เทรนโมเดลของเราเอง
kaggle/rsna-knee-base-probe/  รัน pipeline สำหรับส่งบนเคส gold เพื่อวัดผล
kaggle/rsna-knee-m448-blend/  ลองผสม reader สาธารณะอีกตัวเข้า pipeline 0.943 (4 ต.ค., ได้ 0.942 จึงเลิก)
kaggle/push_retry.sh          push kernel ซ้ำจนกว่า session จะเริ่มจริง
kaggle/watch_jobs.py          หน้าดูสถานะงานและ submission ใน terminal
.vscode/tasks.json            เปิดโฟลเดอร์ใน VS Code แล้วหน้าดูสถานะงานขึ้นใน Terminal เอง
research/                     calibration ของ label, การทดลองหัว, สคริปต์ validate
runs/LOG.md                   บันทึกทุกรอบ: ตั้งค่า, ผล validate, เวลา, คะแนน
docs/                         รายงาน validate (5 ต.ค.) และสรุปงานวิจัยที่เกี่ยวข้อง (9 ต.ค.)
```

## วิธีรัน

สคริปต์ `build_*.py` สร้างโฟลเดอร์ kernel ของ Kaggle จากไฟล์ในนี้ แล้ว push ด้วย Kaggle CLI ตัวอย่าง:

```bash
python kaggle/rsna-knee-gpjw/build_train.py --user <kaggle-user> --suffix r1 --job ctrl,3,auto,256
bash kaggle/push_retry.sh kaggle/rsna-knee-gpjw-train-r1 NvidiaTeslaT4 6
```

ทดสอบในเครื่องโดยไม่ต้องมี GPU หรือข้อมูลจริง (ใช้ DICOM สังเคราะห์):

```bash
python kaggle/rsna-knee-gpjw/smoke_gpjw.py
python kaggle/rsna-knee-raptor/smoke_feats.py
```

ต้องการ Python 3.12, torch, timm, pydicom, pandas, scikit-learn, scipy, opencv-python-headless

สิ่งที่ไม่ได้อยู่ใน repo นี้และต้องเตรียมเอง: ข้อมูลการแข่งขัน, ตาราง label ราย study, น้ำหนักโมเดล และสำเนา notebook สาธารณะที่สคริปต์ `build_*` อ่าน (วางไว้ที่ `research/public_notebooks/`)

## งานสาธารณะที่ใช้ต่อยอด

- inference stack สาธารณะที่ได้ 0.943 และ notebook "Stack + 2.5D ConvNeXt MIL" (0.944)
- สูตรและ checkpoint ของ 2.5D ConvNeXt reader โดย `goodpjw2008`
- โมเดล CoAtNet ตัวเดียวและ notebook 0.949 โดย `nartaa` ซึ่งผู้เขียนเทรนโดยใช้ข้อมูลภายนอก OAI ร่วมด้วย เงื่อนไขการใช้อยู่ในหน้า dataset ของผู้เขียน
- ตาราง label จาก report ที่เผยแพร่สาธารณะ (`pilkwang`, `stevenleehans` และชุดอื่นที่ระบุใน `make_labels_v0.py`)
- โค้ด preprocessing จาก baseline ของ `pilkwang` (Apache-2.0) ที่เส้นทางของโมเดลเราเองเรียกใช้
