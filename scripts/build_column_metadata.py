"""Sinh metadata từng cột cho một dataset, ở đúng định dạng hệ thống đang đọc.

Xuất ra HAI file từ cùng một nguồn:

    <ten>_metadata.json         định dạng {dataset_name, description, columns[]} mà
                                api/services/metadata_gate.py đọc — thả vào gốc repo là
                                cổng metadata sẽ tự bơm cho đúng dataset này.
    <ten>_metadata_review.csv   bản cho người làm nghiệp vụ soát: mỗi dòng một cột, có
                                sẵn ô trống để điền mô tả chính thức.

Vòng lặp xác nhận, chạy lại được nhiều lần:

    1. python scripts/build_column_metadata.py Churn_VT.csv
    2. gửi file _review.csv cho nghiệp vụ, họ điền cột `mo_ta_nghiep_vu_xac_nhan`
    3. chạy lại đúng lệnh ở bước 1 — script đọc lại file review, ưu tiên mô tả đã xác
       nhận, và chỉ giữ suy đoán của máy ở những cột chưa ai điền.

Nguyên tắc: phần THỐNG KÊ là đo trực tiếp; phần MÔ TẢ là suy đoán từ quy ước đặt tên và
được đánh dấu độ tin cậy. Không trộn hai thứ đó vào nhau.

KHÔNG xuất giá trị thô của bản ghi. Định dạng metadata gốc lưu `sample` lấy từ dòng đầu
tiên; với một tập khách hàng thật thì đó là dữ liệu của một người cụ thể (mã thuê bao, mức
cước). Ở đây `sample` là TRUNG VỊ — một số tổng hợp, không truy ngược về ai được.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

#: (mẫu tên cột, nhóm nghiệp vụ, mô tả suy đoán, độ tin cậy)
#: Độ tin cậy "thấp"/"không" là tín hiệu để script đánh dấu cột đó CẦN XÁC NHẬN.
NAMING_RULES: list[tuple[str, str, str, str]] = [
    (r"^OBJID$", "Định danh", "Mã định danh thuê bao.", "cao"),
    (r"^FILTER_(MONTH|YEAR)$", "Bộ lọc", "Tháng/năm của kỳ trích xuất dữ liệu — metadata của lần trích, không phải hành vi khách hàng.", "cao"),
    (r"^fee_total$", "Cước phí", "Tổng cước phí trong kỳ quan sát.", "cao"),
    (r"^fee_avg$", "Cước phí", "Cước phí trung bình mỗi tháng.", "cao"),
    (r"^fee_max$", "Cước phí", "Cước phí tháng cao nhất trong kỳ.", "cao"),
    (r"^fee_std$", "Cước phí", "Độ lệch chuẩn cước phí giữa các tháng.", "cao"),
    (r"^fee_old$", "Cước phí", "Cước phí ở giai đoạn ĐẦU kỳ quan sát.", "cao"),
    (r"^fee_recent$", "Cước phí", "Cước phí ở giai đoạn CUỐI kỳ quan sát.", "cao"),
    (r"^fee_trend$", "Cước phí", "Xu hướng cước phí qua các tháng.", "cao"),
    (r"^active_fee_months$", "Cước phí", "Số tháng có phát sinh cước trong kỳ.", "cao"),
    (r"^no_fee_all_period$", "Cước phí", "Cờ: không phát sinh cước trong suốt kỳ.", "cao"),
    (r"^fee_(Decrease|Increase|Stable)$", "Cước phí", "Cờ phân loại xu hướng cước: giảm / tăng / ổn định.", "cao"),
    (r"^segment_(avg|std)$", "Phân khúc", "Giá trị phân khúc trung bình / độ lệch chuẩn qua các tháng. CÁCH TÍNH PHÂN KHÚC CHƯA RÕ.", "thấp"),
    (r"^high_spender$", "Phân khúc", "Cờ khách hàng chi tiêu cao. NGƯỠNG PHÂN LOẠI CHƯA RÕ.", "thấp"),
    (r"^cl_total_4m$", "Sự cố kỹ thuật", "Tổng số yêu cầu hỗ trợ kỹ thuật (CL) trong 4 tháng.", "trung bình"),
    (r"^cl_avg_4m$", "Sự cố kỹ thuật", "Trung bình số yêu cầu hỗ trợ kỹ thuật (CL) mỗi tháng, trên 4 tháng.", "trung bình"),
    (r"^cl_std$", "Sự cố kỹ thuật", "Độ lệch chuẩn số yêu cầu hỗ trợ kỹ thuật giữa các tháng.", "trung bình"),
    (r"^old_cl$", "Sự cố kỹ thuật", "Số sự cố kỹ thuật ở giai đoạn ĐẦU kỳ.", "cao"),
    (r"^recent_cl$", "Sự cố kỹ thuật", "Số sự cố kỹ thuật ở giai đoạn CUỐI kỳ.", "cao"),
    (r"^active_cl_months$", "Sự cố kỹ thuật", "Số tháng có phát sinh sự cố kỹ thuật.", "cao"),
    (r"^cl_recent_only$", "Sự cố kỹ thuật", "Cờ: sự cố kỹ thuật CHỈ xuất hiện ở giai đoạn cuối kỳ.", "trung bình"),
    (r"^persistent_cl$", "Sự cố kỹ thuật", "Cờ: sự cố kỹ thuật kéo dài qua nhiều tháng.", "trung bình"),
    (r"^no_cl_all_period$", "Sự cố kỹ thuật", "Cờ: không có sự cố kỹ thuật nào trong suốt kỳ.", "cao"),
    (r"^frequent_cl$", "Sự cố kỹ thuật", "Cờ: tần suất sự cố kỹ thuật cao. NGƯỠNG CHƯA RÕ.", "thấp"),
    (r"^escalating_cl$", "Sự cố kỹ thuật", "Cờ: sự cố kỹ thuật tăng dần theo thời gian.", "trung bình"),
    (r"^declining_cl$", "Sự cố kỹ thuật", "Cờ: sự cố kỹ thuật giảm dần theo thời gian.", "trung bình"),
    (r"^complaint_total_6m$", "Khiếu nại", "Tổng số khiếu nại trong 6 tháng.", "trung bình"),
    (r"^complaint_avg_6m$", "Khiếu nại", "Trung bình số khiếu nại mỗi tháng, trên 6 tháng.", "trung bình"),
    (r"^complaint_std$", "Khiếu nại", "Độ lệch chuẩn số khiếu nại giữa các tháng.", "trung bình"),
    (r"^old_complaint$", "Khiếu nại", "Số khiếu nại ở giai đoạn ĐẦU kỳ.", "cao"),
    (r"^recent_complaint$", "Khiếu nại", "Số khiếu nại ở giai đoạn CUỐI kỳ.", "cao"),
    (r"^active_complaint_months$", "Khiếu nại", "Số tháng có phát sinh khiếu nại.", "cao"),
    (r"^complaint_recent_only$", "Khiếu nại", "Cờ: khiếu nại CHỈ xuất hiện ở giai đoạn cuối kỳ.", "trung bình"),
    (r"^persistent_complaint$", "Khiếu nại", "Cờ: khiếu nại kéo dài qua nhiều tháng.", "trung bình"),
    (r"^no_complaint_all_period$", "Khiếu nại", "Cờ: không có khiếu nại nào trong suốt kỳ.", "cao"),
    (r"^escalating_complaint$", "Khiếu nại", "Cờ: khiếu nại tăng dần theo thời gian.", "trung bình"),
    (r"^declining_complaint$", "Khiếu nại", "Cờ: khiếu nại giảm dần theo thời gian.", "trung bình"),
    (r"^downtime_(avg|max|min|std)$", "Gián đoạn dịch vụ", "Thời gian gián đoạn dịch vụ (trung bình/lớn nhất/nhỏ nhất/độ lệch chuẩn). ĐƠN VỊ CHƯA RÕ — phút, giờ hay ngày?", "không"),
    (r"^downtime_trend$", "Gián đoạn dịch vụ", "Xu hướng thời gian gián đoạn qua các tháng. ĐƠN VỊ CHƯA RÕ.", "không"),
    (r"^(positive|stable|negative|severe_negative)_months$", "Chất lượng đường truyền", "Số tháng ở trạng thái tốt / ổn định / xấu / rất xấu. TIÊU CHÍ PHÂN LOẠI CHƯA RÕ — dựa trên chỉ số nào và ngưỡng bao nhiêu?", "không"),
    (r"^ever_downtrend$", "Chất lượng đường truyền", "Cờ: từng ghi nhận xu hướng đi xuống.", "trung bình"),
    (r"^persistent_negative$", "Chất lượng đường truyền", "Cờ: trạng thái xấu kéo dài qua nhiều tháng.", "trung bình"),
    (r"^branch_(std|avg|trend_slope)$", "Chi nhánh", "Chỉ số theo chi nhánh (độ lệch chuẩn / trung bình / độ dốc xu hướng). Ý NGHĨA CHƯA RÕ — đây là chỉ số của CHI NHÁNH quản lý, hay chỉ số của thuê bao gộp theo chi nhánh?", "không"),
    (r"^usage_trend_(avg|max|min|std|trend)$", "Mức sử dụng", "Xu hướng mức sử dụng (trung bình/lớn nhất/nhỏ nhất/độ lệch chuẩn/xu hướng của xu hướng). ĐƠN VỊ VÀ CÁCH TÍNH CHƯA RÕ.", "không"),
    (r"^usage_(positive|stable|negative)_months$", "Mức sử dụng", "Số tháng mức sử dụng tăng / ổn định / giảm.", "trung bình"),
    (r"^HSSD$", "Chưa rõ", "VIẾT TẮT CHƯA GIẢI MÃ ĐƯỢC. Cần nghiệp vụ cho biết tên đầy đủ và cách tính.", "không"),
    (r"^CTBDV$", "Chưa rõ", "VIẾT TẮT CHƯA GIẢI MÃ ĐƯỢC. Cần nghiệp vụ cho biết tên đầy đủ và cách tính.", "không"),
    (r"^LLSD_\d{6}$", "Chưa rõ", "VIẾT TẮT CHƯA GIẢI MÃ ĐƯỢC; hậu tố YYYYMM cho thấy là chỉ số theo tháng. Cần tên đầy đủ.", "không"),
    (r"^HTKT_Net_\d{6}$", "Hỗ trợ kỹ thuật", "Nhiều khả năng chỉ số hỗ trợ kỹ thuật mảng Net theo tháng YYYYMM. Cần xác nhận cách tính.", "thấp"),
    (r"^HTKT_CHECKLIST_\d{6}$", "Hỗ trợ kỹ thuật", "Nhiều khả năng số checklist hỗ trợ kỹ thuật trong tháng YYYYMM. Cần xác nhận.", "thấp"),
    (r"^CHECKLIST_DUPLICATED_\d{6}$", "Hỗ trợ kỹ thuật", "Nhiều khả năng số checklist bị trùng lặp trong tháng YYYYMM. Cần xác nhận 'trùng lặp' được định nghĩa thế nào.", "thấp"),
    (r"^total_negative_\d{6}$", "Chất lượng đường truyền", "Nhiều khả năng tổng số chỉ số âm/xấu trong tháng YYYYMM. Cần xác nhận 'negative' đo trên chỉ số nào.", "thấp"),
    (r"^total_call_\d+d$", "Cuộc gọi", "Tổng số cuộc gọi trong N ngày gần nhất. Cần xác nhận: gọi ĐẾN tổng đài hay gọi ĐI?", "thấp"),
    (r"^total_missed_\d+d$", "Cuộc gọi", "Tổng số cuộc gọi nhỡ trong N ngày gần nhất. Cần xác nhận nhỡ ở phía nào.", "thấp"),
    (r"^ratio_missed_\d+d$", "Cuộc gọi", "Tỉ lệ cuộc gọi nhỡ trên tổng cuộc gọi trong N ngày gần nhất.", "trung bình"),
    (r"^AMT_ACTIVE_DAY_M01$", "Chưa rõ", "Nhiều khả năng số ngày hoạt động trong tháng M01. BẤT THƯỜNG: chỉ có duy nhất M01 trong khi các họ cột khác đều có 4-6 tháng — thiếu cột hay có chủ đích?", "không"),
]

NEEDS_REVIEW = {"thấp", "không"}
REVIEW_COLUMN = "mo_ta_nghiep_vu_xac_nhan"

#: Tên cột "tên cột" và "mô tả" mà file review có thể quay về dưới nhiều dạng — xem
#: load_confirmations().
_NAME_HEADERS = ("cot", "Cột", "column", "Column")
_DESCRIPTION_HEADERS = ("mo_ta_suy_doan_cua_may", "Mô tả", "mo_ta", "description")


def infer(column: str) -> tuple[str, str, str]:
    """Trả (nhóm, mô tả suy đoán, độ tin cậy) cho một tên cột."""
    for pattern, group, description, confidence in NAMING_RULES:
        if re.match(pattern, column):
            return group, description, confidence
    return "Chưa phân loại", "Không khớp quy ước đặt tên nào đã biết. Cần nghiệp vụ mô tả.", "không"


def _first_header(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((h for h in candidates if h in frame.columns), None)


def load_confirmations(review_path: Path) -> dict[str, str]:
    """Mô tả nghiệp vụ đã xác nhận, đọc từ file review quay về.

    Người làm nghiệp vụ soát bằng Excel/Google Sheets, nên file quay về hiếm khi giữ
    nguyên hình dạng đã gửi đi. Bản Churn_VT quay về đã: đổi tiêu đề cột sang tiếng Việt
    có dấu (`cot` → `Cột`, `mo_ta_suy_doan_cua_may` → `Mô tả`), bỏ hẳn cột `do_tin_cay`,
    xoá dòng OBJID, và — điểm quan trọng nhất — SỬA THẲNG vào cột mô tả thay vì điền vào
    ô trống dành riêng. Bản đọc cũ chỉ nhìn đúng hai tiêu đề gốc nên nhận về 0 xác nhận
    trong khi file có 52.

    Vì vậy: chấp nhận nhiều biến thể tiêu đề, và coi một mô tả là ĐÃ XÁC NHẬN khi nó khác
    với suy đoán hiện tại của máy cho đúng cột đó. infer() là hàm thuần nên phép so sánh
    này ổn định qua các lần chạy lại.
    """
    if not review_path.exists():
        return {}
    previous = pd.read_csv(review_path)
    name_header = _first_header(previous, _NAME_HEADERS)
    if name_header is None:
        return {}

    def cleaned(header: str | None) -> dict[str, str]:
        if header is None:
            return {}
        filled = previous[previous[header].notna() & (previous[header].astype(str).str.strip() != "")]
        return dict(zip(filled[name_header].astype(str), filled[header].astype(str).str.strip()))

    # Ô trống dành riêng thắng, vì điền vào đó là hành động rõ ràng nhất; mô tả bị sửa
    # thẳng chỉ tính khi nó lệch khỏi suy đoán của máy.
    confirmed = {
        name: text for name, text in cleaned(_first_header(previous, _DESCRIPTION_HEADERS)).items()
        if text != infer(name)[1]
    }
    confirmed.update(cleaned(REVIEW_COLUMN if REVIEW_COLUMN in previous.columns else None))
    return confirmed


def load_group_overrides(review_path: Path) -> dict[str, str]:
    """Nhóm nghiệp vụ đã sửa lại trong file review.

    Cùng cơ chế với load_confirmations(): nhóm nào khác suy đoán của máy là nhóm đã được
    sửa có chủ đích. Trên bản Churn_VT quay về có 16 cột bị xếp lại nhóm, và chúng sửa
    những chỗ máy đoán SAI HẲN chứ không phải đổi cách gọi: `branch_*` không phải chỉ số
    chi nhánh mà là thống kê lưu lượng (LLSD), `total_negative_*` là điểm chạm tiêu cực
    chứ không phải chỉ số chất lượng đường truyền.
    """
    if not review_path.exists():
        return {}
    previous = pd.read_csv(review_path)
    name_header = _first_header(previous, _NAME_HEADERS)
    group_header = _first_header(previous, ("nhom", "Nhóm", "group"))
    if name_header is None or group_header is None:
        return {}
    filled = previous[previous[group_header].notna() & (previous[group_header].astype(str).str.strip() != "")]
    return {
        str(name): str(group).strip()
        for name, group in zip(filled[name_header], filled[group_header])
        if str(group).strip() != infer(str(name))[0]
    }


def build(csv_path: Path, returned_review: Path | None = None) -> tuple[Path, Path, int, int]:
    stem = csv_path.stem
    json_path = csv_path.with_name(f"{stem}_metadata.json")
    review_path = csv_path.with_name(f"{stem}_metadata_review.csv")

    # File quay về thường KHÔNG còn tên cũ — Google Sheets xuất ra
    # "Churn_VT_metadata_review - Churn_VT_metadata_review.csv". Đọc xác nhận từ đó,
    # nhưng vẫn ghi đè bản chuẩn ở review_path để vòng sau gửi đi từ một chỗ duy nhất.
    source_of_truth = returned_review or review_path
    confirmed = load_confirmations(source_of_truth)
    regrouped = load_group_overrides(source_of_truth)
    df = pd.read_csv(csv_path, low_memory=False)
    total_rows = len(df)

    columns, review_rows = [], []
    for name in df.columns:
        series = df[name]
        group, guess, confidence = infer(str(name))
        group = regrouped.get(name, group)
        description = confirmed.get(name, guess)
        is_confirmed = name in confirmed
        numeric = pd.api.types.is_numeric_dtype(series)
        non_null = int(series.notna().sum())

        # `sample` là TRUNG VỊ, không phải giá trị của một bản ghi cụ thể — xem docstring.
        if numeric and non_null:
            sample = f"{float(pd.to_numeric(series, errors='coerce').median()):g}"
        else:
            sample = ""

        columns.append({
            "column": str(name),
            "type": str(series.dtype),
            "sample": sample,
            "description": description,
            "group": group,
            "confirmed": is_confirmed,
        })
        review_rows.append({
            "cot": str(name),
            "nhom": group,
            "mo_ta_suy_doan_cua_may": guess,
            "do_tin_cay": confidence,
            "can_xac_nhan": "" if is_confirmed else ("CÓ" if confidence in NEEDS_REVIEW else ""),
            "kieu_du_lieu": str(series.dtype),
            "ti_le_thieu_pct": round((total_rows - non_null) / total_rows * 100, 2),
            "so_gia_tri_khac_nhau": int(series.nunique(dropna=True)),
            "trung_vi": sample,
            REVIEW_COLUMN: confirmed.get(name, ""),
        })

    json_path.write_text(json.dumps({
        "dataset_name": stem,
        "description": (
            f"Metadata từng cột cho {csv_path.name} ({total_rows:,} dòng, {len(df.columns)} cột). "
            "Mô tả có `confirmed: true` là do nghiệp vụ xác nhận; còn lại là suy đoán từ quy ước "
            "đặt tên, chưa được xác nhận."
        ),
        "columns": columns,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(review_rows).to_csv(review_path, index=False, encoding="utf-8-sig")
    return json_path, review_path, len(columns), sum(1 for r in review_rows if r["can_xac_nhan"] == "CÓ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path, help="đường dẫn tới file CSV nguồn")
    parser.add_argument("--review", type=Path, default=None,
                        help="file review nghiệp vụ gửi về, nếu đã bị đổi tên khi tải xuống")
    args = parser.parse_args()
    for path in (args.csv_path, args.review):
        if path is not None and not path.exists():
            print(f"không thấy file: {path}", file=sys.stderr)
            return 1

    json_path, review_path, n_cols, n_review = build(args.csv_path, args.review)
    print(f"{json_path.name}   {n_cols} cột — định dạng hệ thống đọc được")
    print(f"{review_path.name}   gửi nghiệp vụ, {n_review} cột cần xác nhận")
    print(f"\nSau khi nghiệp vụ điền cột '{REVIEW_COLUMN}', chạy lại đúng lệnh này "
          f"để nạp mô tả đã xác nhận vào JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
