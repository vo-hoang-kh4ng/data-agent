# Câu hỏi gửi nghiệp vụ — bản trích xuất `final_churn_VT.csv`

62.467 dòng · 97 cột · đã nhận 52/97 mô tả xác nhận (cảm ơn anh/chị)

Mỗi câu hỏi bên dưới đều kèm con số đo trực tiếp trên file, để anh/chị đối chiếu nhanh.
Trả lời được câu nào thì điền câu đó, không cần đủ hết.

---

## A. 14 cột chưa có mô tả

Đây là các cột còn để trống trong file review (`Churn_VT_metadata_review.csv`, cột
`mo_ta_nghiep_vu_xac_nhan`). Tất cả đều gần như đủ dữ liệu, nên chỉ thiếu ý nghĩa.

| Cột | Đã đo được | Cần biết |
|---|---|---|
| `segment_avg` | 17 giá trị khác nhau, trung vị 2 | Phân khúc được tính thế nào, thang mấy bậc? |
| `segment_std` | 19 giá trị, trung vị 0 | Độ lệch chuẩn của phân khúc qua mấy tháng? |
| `high_spender` | cờ 0/1 | Ngưỡng nào thì được tính là chi tiêu cao? |
| `frequent_cl` | **toàn bộ 62.467 dòng đều = 1** | Cột này có bị lỗi khi trích xuất không? Nếu mọi thuê bao đều "frequent" thì cờ không phân biệt được ai. |
| `downtime_trend` | 32 giá trị, trung vị 0 | Xu hướng tính trên mấy tháng, đơn vị là gì? |
| `usage_trend_avg` | 17 giá trị, trung vị −0,5 | |
| `usage_trend_max` | 5 giá trị | |
| `usage_trend_min` | 5 giá trị, trung vị −2 | Cả 5 cột `usage_trend_*`: đo trên chỉ số nào (LLSD?), đơn vị gì, và "trend của trend" (`usage_trend_trend`) nghĩa là gì? |
| `usage_trend_std` | 20 giá trị | |
| `usage_trend_trend` | 33 giá trị | |
| `HTKT_Net_202603` | thiếu 2,00% | |
| `HTKT_Net_202604` | thiếu 0,43% | 4 cột `HTKT_Net_*`: khác gì với `HTKT_CHECKLIST_*` cùng tháng? |
| `HTKT_Net_202605` | thiếu 0,04% | |
| `HTKT_Net_202606` | đủ dữ liệu, 9 giá trị | |

---

## B. 3 chỗ trong bản đã trả lời, nhờ anh/chị xem lại

**B1. `total_missed_90d` đang ghi là "Tỉ lệ cuộc gọi nhỡ trong 90 ngày"** — trùng y hệt mô
tả của `ratio_missed_90d`. Nhưng số liệu cho thấy nó là **số đếm**, không phải tỉ lệ: giá
trị nguyên từ 2 đến 58, trong khi `ratio_missed_90d` chạy từ 0,05 đến 1,0. Nhiều khả năng
bị copy nhầm dòng. Mô tả đúng có phải là "Tổng số cuộc gọi đi thất bại trong 90 ngày"?

**B2. `HTKT_CHECKLIST_260329`** — anh/chị trả lời "Không rõ". Hậu tố `260329` không theo
định dạng `YYYYMM` như các cột cùng họ (`202603`…`202606`). Chỉ 298/62.467 dòng khác 0,
giá trị lớn nhất là 2. Cột này có phải sinh nhầm khi trích xuất không? Nếu đúng thì bỏ.

**B3. Nhóm `downtime_*` (avg/max/min/std/trend)** — phần "ĐƠN VỊ CHƯA RÕ — phút, giờ hay
ngày?" vẫn còn nguyên trong bản gửi về. Đây là 5 cột, nhờ anh/chị cho biết đơn vị.

**B4. Hai cờ sự cố kỹ thuật gần như rỗng.** `persistent_cl` ("sự cố kéo dài qua nhiều
tháng") chỉ có **2/62.467** dòng bằng 1; `persistent_negative` có 157 dòng. Với 62 nghìn
thuê bao đã rời mạng, con số này thấp bất thường. Ngưỡng để tính là "kéo dài" là bao nhiêu
tháng — hay cờ bị tính sai khi trích xuất?

---

## C. Câu hỏi mới, phát sinh khi phân tích

### C1. Ô TRỐNG nghĩa là "không phát sinh" hay "không đo"? *(quan trọng nhất)*

Đây là câu hỏi có ảnh hưởng lớn nhất tới kết quả. Với một cột để trống, hệ thống có hai
cách hiểu hoàn toàn khác nhau:

- **"không phát sinh"** → điền 0, và thuê bao đó được tính là *có dữ liệu, giá trị 0*
- **"không đo/không thuộc phạm vi"** → phải loại ra, vì điền 0 là **bịa** một con số

Hiện tại chúng tôi đang **loại** các cột này, vì có một bằng chứng nghiêng về hướng đó:
**số 0 đã được ghi nhận tường minh ở những dòng có dữ liệu**. Ví dụ `total_negative_202601`
có 7.751 dòng có giá trị, trong đó min = 0 — tức khi thật sự không phát sinh thì hệ thống
*có* ghi số 0. Vậy ô trống nhiều khả năng mang nghĩa khác.

Nhờ anh/chị xác nhận cho từng họ cột:

| Họ cột | Tỉ lệ trống | Ô trống nghĩa là gì? |
|---|---|---|
| `total_negative_202601…202606` | 67,7% – 90,5% | ☐ không phát sinh điểm chạm ☐ tháng đó không đo ☐ khác: |
| `LLSD_202603…202606` | 13,1% – 61,4% | ☐ không phát sinh lưu lượng ☐ không đo ☐ khác: |
| `total_call_*`, `total_missed_*`, `ratio_missed_*` | 98,2% | ☐ không có cuộc gọi ☐ không đo ☐ khác: |

Nếu là "không phát sinh", chúng tôi lấy lại được 6 cột điểm chạm tiêu cực vào phân tích —
đây là tín hiệu có giá trị, hiện đang phải bỏ.

### C2. Nhóm cột cuộc gọi chỉ có 1.133/62.467 dòng (1,8%) — và không dòng nào có giá trị 1

Cả 9 cột `total_call_*` / `total_missed_*` / `ratio_missed_*` có dữ liệu ở **đúng cùng
1.133 dòng**. Hai điều cần hỏi:

1. **1,8% này là ai?** Có phải một tập được lọc riêng (ví dụ chỉ thuê bao có phát sinh sự
   cố) không? Nếu có tiêu chí lọc, xin cho biết — mọi thống kê trên các cột này chỉ đại
   diện cho nhóm đó, không đại diện cho 62.467 thuê bao.

2. **Giá trị 1 không xuất hiện ở bất kỳ cột nào trong 9 cột.** `total_call_90d` chạy từ 2
   đến 308, không có 0 và không có 1. `total_call_30d` thì có 1.057 dòng bằng 0, nhưng
   cũng không dòng nào bằng 1. Đây là hiện tượng bất thường — có phải điều kiện lọc là
   "≥ 2 cuộc gọi" không, hay cách đếm gộp theo cặp?

### C3. `AMT_ACTIVE_DAY_M01` — tên cột và mô tả đang nói hai thứ khác nhau

Bản mới đã có dữ liệu (62.461/62.467 dòng, trước đây trống 100%). Nhưng:

- **tên cột** ghi `ACTIVE_DAY` → ngày
- **mô tả anh/chị gửi** ghi "Số **tháng** sử dụng Internet"
- **giá trị** có phần thập phân (trung vị 25,1) và lớn nhất là **200,5**

Nếu là tháng thì 200,5 tháng ≈ 16,7 năm — hợp lý cho thâm niên. Nếu là ngày thì hậu tố
`M01` không còn ý nghĩa vì vượt xa 31. Nhờ anh/chị chốt: **đơn vị là ngày hay tháng**, và
`M01` chỉ điều gì?

Cột này quan trọng vì hiện là biến duy nhất có thể đại diện **thâm niên thuê bao**.

### C4. `HSSD` / `CTBDV` có phải mã lý do huỷ chính thức từ hệ thống không?

Chúng tôi đo được:

| | Số thuê bao | Tỉ lệ |
|---|---|---|
| `HSSD` = 1 (Hủy sau sử dụng) | 30.263 | 48,4% |
| `CTBDV` = 1 (Chủ thuê bao đi vắng) | 27.671 | 44,3% |
| cả hai = 0 | 4.533 | 7,3% |

Hai cờ **loại trừ nhau tuyệt đối** — không có dòng nào cùng bằng 1.

Anh/chị đã cho biết nhóm "cả hai = 0" là các thuê bao **đã khôi phục dịch vụ**. Xin xác
nhận thêm hai điểm:

1. Hai cột này có phải **mã lý do huỷ lấy từ hệ thống** (đã ghi nhận sẵn), hay là do đội
   phân tích suy ra từ hành vi? Điều này quyết định báo cáo có được phép **nêu lý do rời
   mạng** hay chỉ được **mô tả đặc điểm**.
2. Ngoài `HSSD` và `CTBDV` còn mã lý do huỷ nào khác không? 92,7% có mã, phần còn lại là
   nhóm khôi phục — nếu danh mục mã đầy đủ có nhiều hơn 2 giá trị thì bản trích xuất này
   đang thiếu.

---

## Cách gửi lại

Điền trực tiếp vào file `Churn_VT_metadata_review.csv` (cột `mo_ta_nghiep_vu_xac_nhan`),
hoặc trả lời thẳng trong file này cũng được — chúng tôi đọc được cả hai định dạng, kể cả
khi anh/chị sửa đè lên cột mô tả hoặc đổi tên tiêu đề cột.
