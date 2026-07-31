"""Chuẩn bị file Churn_VT cho pipeline persona: dựng cột trạng thái từ mã huỷ.

Vì sao là script riêng chứ không nằm trong pipeline: `HSSD` và `CTBDV` chỉ có nghĩa với
đúng bản trích xuất này. Nhánh hiện tại tồn tại để GỠ tên cột telco ra khỏi đường đi
chung, nên pipeline chỉ biết đếm một cột trạng thái do caller đưa vào, không biết HSSD là
gì và không được phép biết.

Nghiệp vụ đã xác nhận, và crosstab trên 62.467 dòng khớp:

    HSSD  = 1   Hủy sau sử dụng          30.263   (48,4%)
    CTBDV = 1   Chủ thuê bao đi vắng     27.671   (44,3%)
    cả hai = 0  đã khôi phục dịch vụ      4.533   ( 7,3%)

4.533 dòng cuối chính là chỗ bảng điều khiển đang sai: nó in cứng "100% — Toàn bộ mẫu đã
rời mạng" cho mọi dataset POST_CHURN, không đo gì cả. Sai 4.533 người, và sai về đúng phía
làm tắt kịch bản giữ chân cho nhóm DUY NHẤT còn giữ chân được.

Cách dùng:

    python scripts/prepare_churn_vt.py final_churn_VT.csv
    # -> final_churn_VT_prepared.csv, thêm cột `trang_thai`

rồi truyền vào pipeline:

    run_persona_pipeline(data, status_col=STATUS_COLUMN,
                         active_status_values=ACTIVE_STATUS_VALUES)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: Tên cột trạng thái mà script này sinh ra.
STATUS_COLUMN = "trang_thai"

CANCELLED_AFTER_USE = "Hủy sau sử dụng"
SUBSCRIBER_AWAY = "Chủ thuê bao đi vắng"
RESTORED = "Đã khôi phục dịch vụ"

#: Giá trị nào nghĩa là thuê bao VẪN CÒN là khách hàng. Truyền thẳng vào
#: run_persona_pipeline(active_status_values=...) — pipeline không tự suy ra được.
ACTIVE_STATUS_VALUES = {RESTORED}


def derive_status(data: pd.DataFrame) -> pd.Series:
    """Dựng cột trạng thái từ hai mã huỷ.

    Tính loại trừ lẫn nhau của HSSD/CTBDV là giả định mà mọi thứ còn lại dựa lên, nên nó
    được KIỂM TRA chứ không được tin: nếu một bản trích xuất sau này phá vỡ điều đó, dừng
    hẳn còn hơn âm thầm xếp những dòng ấy vào nhóm đã khôi phục.

    Args:
        data: Khung dữ liệu Churn_VT, phải có cả `HSSD` và `CTBDV`.

    Returns:
        Series nhãn trạng thái, giữ NaN ở những dòng không có mã — thiếu mã KHÔNG phải
        bằng chứng thuê bao đã khôi phục.

    Raises:
        ValueError: Thiếu cột mã, hoặc có dòng mang cả hai mã.
    """
    missing = [c for c in ("HSSD", "CTBDV") if c not in data.columns]
    if missing:
        raise ValueError(
            f"thiếu cột mã huỷ: {', '.join(missing)} — không dựng được cột trạng thái"
        )

    hssd = pd.to_numeric(data["HSSD"], errors="coerce")
    ctbdv = pd.to_numeric(data["CTBDV"], errors="coerce")

    both = int(((hssd == 1) & (ctbdv == 1)).sum())
    if both:
        raise ValueError(
            f"{both} dòng mang CẢ HAI mã HSSD và CTBDV — hai mã này phải loại trừ nhau. "
            "Giả định của toàn bộ phép dựng trạng thái không còn đúng với bản trích xuất này."
        )

    status = pd.Series(pd.NA, index=data.index, dtype="object")
    status[hssd == 1] = CANCELLED_AFTER_USE
    status[ctbdv == 1] = SUBSCRIBER_AWAY
    status[(hssd == 0) & (ctbdv == 0)] = RESTORED
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path, help="đường dẫn tới file Churn_VT")
    parser.add_argument("--out", type=Path, default=None, help="file đích (mặc định: <ten>_prepared.csv)")
    args = parser.parse_args()
    if not args.csv_path.exists():
        print(f"không thấy file: {args.csv_path}", file=sys.stderr)
        return 1

    data = pd.read_csv(args.csv_path, low_memory=False)
    try:
        data[STATUS_COLUMN] = derive_status(data)
    except ValueError as e:
        print(f"dừng: {e}", file=sys.stderr)
        return 1

    out = args.out or args.csv_path.with_name(f"{args.csv_path.stem}_prepared.csv")
    data.to_csv(out, index=False, encoding="utf-8-sig")

    counts = data[STATUS_COLUMN].value_counts(dropna=False)
    total = len(data)
    print(f"{out.name}   {total:,} dòng, thêm cột `{STATUS_COLUMN}`")
    for label, n in counts.items():
        print(f"  {str(label):24s} {n:7,}  {n / total * 100:5.1f}%")
    print(f"\nTruyền vào pipeline: status_col={STATUS_COLUMN!r}, "
          f"active_status_values={ACTIVE_STATUS_VALUES!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
