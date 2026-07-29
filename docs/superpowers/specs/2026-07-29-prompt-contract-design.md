# Gác mối nối giữa prompt và lõi persona

> Trạng thái: **đã duyệt thiết kế**, chưa triển khai.
> Ngày: 2026-07-29. Nhánh: `fix/persona-generic-telco-hardcodes`.

---

## 1. Vấn đề

Lõi persona không có đường gọi tĩnh nào từ production. Toàn bộ `triadic_dgm/persona/` chỉ
được gọi từ một **chuỗi ký tự** trong `triadic_dgm/prompts/prompts.py`, do LLM sandbox đọc
rồi gõ lại thành code.

```
prompts.py (chuỗi)  →  LLM sandbox viết code  →  import run_persona_pipeline
                    →  in [JSON_START_PERSONA]…[JSON_END_PERSONA]
                    →  extract_persona_list()  →  report_generator
```

Đo được, không phải suy đoán:

- `grep -rn "run_persona_pipeline" --include=*.py .` ngoài `tests/` → **0 kết quả**. Không
  file production nào import nó.
- Thứ duy nhất đang gác là `tests/test_prompt_invariant.py:40`:
  `assert "run_persona_pipeline" in body`. Đó là phép kiểm tra **chuỗi con trong prompt** —
  nó vẫn xanh kể cả khi hàm đó đã bị xoá khỏi repo.
- `extract_persona_list()` trả `[]` khi không thấy marker và **không bao giờ raise**. Marker
  được giữ ở hai nơi (prompt và parser) mà không gì buộc chúng khớp.

Hai mối nối, cả hai bằng chuỗi, không có gì đối chiếu chúng với nhau:

| # | Mối nối | Gãy thì sao |
|---|---|---|
| 1 | Tên hàm + signature | LLM viết `import` lỗi → vòng sửa lỗi 5 lượt → người dùng không nhận báo cáo |
| 2 | Cặp marker JSON | `extract_persona_list` trả `[]` → báo cáo rỗng, **im lặng**, không có lỗi nào |

Rủi ro là **tương lai**, không phải hiện tại: chạy thử import mọi symbol prompt nêu
(`run_persona_pipeline`, `save_cluster_chart`) — đều còn, signature khớp. Đổi tên một hàm
là gãy runtime mà không test nào đỏ.

## 2. Phạm vi

Mục tiêu đã chốt: **chặn gãy âm thầm**, giữ nguyên kiến trúc. LLM vẫn viết code, vẫn gọi
pipeline.

Ngoài phạm vi, đã cân nhắc và loại:

- **Thêm đường chạy tất định song song** (endpoint Python gọi thẳng pipeline, LLM chỉ diễn
  giải). Đây là thay đổi kiến trúc thật, đụng `api/` và tầng báo cáo. Để riêng.
- **Bỏ hẳn LLM khỏi phân cụm.** Mất khả năng ứng biến với yêu cầu lạ của người dùng.

## 3. Phương án đã chọn — chạy chính khối code trong prompt

Rút các block ` ```python ` trong prompt có nhắc `run_persona_pipeline`, `exec` **nguyên
văn** với dữ liệu tổng hợp, bắt stdout, parse bằng đúng hàm production.

Đã dựng thử trước khi chốt — không thiết kế trên giả định:

```
blocks found: 2
block 0: ok, personas=3, names=['Khách hàng ổn định - Nhóm 1', 'Nhóm spend cao', 'Nhóm visits thấp']
block 1: ok, personas=3, names=[...]
```

### Hai phương án bị loại

**A — chỉ kiểm symbol + signature tĩnh.** Nhanh hơn, nhưng bỏ sót hai lớp: lệch marker, và
lỗi lúc chạy. Cụ thể `json.dumps(personas)` sẽ ném exception nếu pipeline lỡ trả về
`numpy.int64`, và phép kiểm tĩnh không thấy gì.

**C — đưa marker và dòng import thành hằng số Python, prompt nội suy vào.** Về lý thuyết
làm lệch trở nên bất khả. Loại vì `LAMBDA.py:93` gọi `.format(working_path=…)` trên prompt,
nên toàn bộ chuỗi 300 dòng phải nhân đôi ngoặc (`{{`/`}}`) — một bất biến đang có test gác.
Thêm placeholder là thêm rủi ro vào chính bất biến đó, đổi lấy thứ mà phương án đã chọn đã
bắt được end-to-end với rủi ro bằng không.

## 4. Thiết kế

### 4.1 `tests/test_prompt_contract.py` (mới)

Một file test, **không đụng code production**.

| Hạng mục | Nội dung |
|---|---|
| Rút block | Regex ` ```python\n(.*?)``` ` trên thân prompt, lọc block có `run_persona_pipeline` |
| Dữ liệu | DataFrame tổng hợp 300 dòng, 3 nhóm tách được rõ, seed cố định. Không đụng `data/` |
| Thực thi | `exec` nguyên văn, namespace chỉ có `data` và `behavioral_features` — đúng những gì prompt giả định sandbox đã có |
| Parse | `triadic_dgm.services.persona_json.extract_persona_list` — **hàm production**, không phải bản sao trong test |

Chạy trên **cả hai dạng prompt**. `LAMBDA.py:93` gọi `.format(working_path=…)`;
`triadic_dgm/agent/programmer.py:159` **cố ý không** format (có comment giải thích ở dòng
145 rằng đã thử và đã revert). Hai call site, hai dạng chuỗi khác nhau, và bất biến ngoặc
nhân đôi chỉ có thể gãy ở dạng có format — nên phải test cả hai.

### 4.2 Các khẳng định

1. Tìm được **ít nhất 1** block — gác việc block bị xoá khỏi prompt.
2. Mỗi block `exec` không ném exception.
3. Stdout của mỗi block parse ra **≥ 2 persona**, mỗi persona có đủ key bắt buộc.
4. `json.dumps` trên kết quả pipeline thành công — gác kiểu không serialize được.
5. Hai dạng prompt (formatted / unformatted) cho cùng số block và cùng kết quả.

### 4.3 Cố ý KHÔNG gác

- **Tên persona cụ thể** — đã là việc của `tests/test_pipeline_naming.py`. Test này khẳng
  định *có persona hợp lệ*, không khẳng định *chúng tên gì*.
- **Những gì LLM ứng biến khi đi chệch khối mẫu.** Không test nào chạm tới được; nói rõ
  trong docstring để người sau không tưởng nhầm là đã phủ.

## 5. Lỗ đặt tên, phát hiện bởi chính prototype

Prototype trả về `'Khách hàng ổn định - Nhóm 1'` trên dữ liệu `spend`/`visits` tổng hợp —
tức từ vựng khách hàng vẫn lọt vào một dataset generic.

Nguyên nhân, `triadic_dgm/persona/pipeline.py:453`:

```python
if new_name and not p["is_anomaly"]:
```

`name_by_top_feature` trả `None` khi không feature nào lệch đủ mạnh, với ý định (ghi trong
docstring) là *"caller keeps whatever it already had"*. Trên đường GENERIC, thứ nó "đã có"
lại là fallback của rule engine — `rules.py:150`, `"Khách hàng ổn định"`, một thang bậc
toàn predicate telco. Cụm nằm giữa, không lệch mạnh ở đâu, luôn rơi vào đó.

Đây là lỗ trong bản sửa đặt tên ngày 2026-07-27, không phải lỗi mới. Chuẩn của chính dự án
coi là rò rỉ: `tests/test_pipeline.py:158` đã assert `"khách hàng" not in persona_name` —
nhưng chỉ cho nhánh thất bại, không cho nhánh thành công.

**Sửa trong đợt này**, vì nếu không thì test hợp đồng mới sẽ đóng băng một kết quả đang rò
rỉ thành "đúng".

Cách sửa: trên đường GENERIC, khi `name_by_top_feature` trả `None`, đặt tên
**`"Nhóm gần trung bình toàn tập"`** thay vì giữ fallback telco. Chọn đúng chuỗi này chứ
không phải "một tên trung tính nào đó": nó mô tả chính điều đã đo được (không feature nào
lệch đủ mạnh), không giả định dataset nói về ai, và không mượn từ vựng của miền nào.
`_dedupe_names` đã có sẵn lo phần trùng tên khi nhiều cụm cùng rơi vào trường hợp này.

Kèm test: một dataset generic mà mọi cụm đều gần trung bình phải không sinh ra chuỗi
`"khách hàng"` nào trong `persona_name`. Đường telco không đổi.

## 6. Không thay đổi

- Không đụng `triadic_dgm/prompts/prompts.py`. Không có thay đổi prompt ⇒ không cần thêm
  mục vào `prompts/CHANGELOG.md`.
- Không đụng `api/`, không đụng tầng báo cáo.
- Đường telco giữ nguyên hoàn toàn.
- Không đụng `data/` (ràng buộc `CLAUDE.md`); dữ liệu test là tổng hợp, seed cố định.

## 7. Tiêu chí hoàn thành

- `pytest tests/` xanh, không cần cờ nào. Hiện tại: 236 passed, 1 skipped.
- Đổi tên `run_persona_pipeline` trong `pipeline.py` mà không sửa prompt ⇒ test hợp đồng
  **đỏ**. Đây là phép thử duy nhất chứng minh việc này có tác dụng, và phải chạy thật để
  xác nhận, không được suy luận.
- Đổi `[JSON_END_PERSONA]` một phía ⇒ test hợp đồng **đỏ**.
- Trên dataset generic mà mọi cụm gần trung bình, không `persona_name` nào chứa
  `"khách hàng"`. Khẳng định này thuộc **test đặt tên** (§5), không phải test hợp đồng —
  test hợp đồng cố ý không phát biểu gì về tên persona (§4.3).
