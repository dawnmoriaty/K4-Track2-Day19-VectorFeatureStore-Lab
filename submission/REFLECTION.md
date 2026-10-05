# Reflection — Lab 19

**Tên:** Phùng Đức Đăng
**Cohort:** A20-K4
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

**1. Kết quả trên 50 golden queries:**
- `exact` (15): **BM25 & Hybrid** thắng (96.7% vs Vector 88.7%). BM25 so khớp từ vựng chính xác tuyệt đối, tránh semantic drift của vector trên từ khóa hiếm/mã lỗi.
- `mixed` (20): **Hybrid** thắng áp đảo (**100.0%** vs BM25 97.0%, Vector 98.5%). RRF ($k=60$) cộng hưởng hoàn hảo giữa neo từ khóa cốt lõi và mở rộng ngữ nghĩa.
- `paraphrase` (15): BM25 đạt 33.3%, Hybrid 32.0%, Vector 24.0% (mô hình embedding nhẹ gặp khó với các sắc thái phức tạp). Tổng thể Hybrid cao nhất (78.6%).

**2. Khi KHÔNG dùng Hybrid:**
- **Pure BM25:** Tra cứu mã lỗi, ID, SKU, ký hiệu code hoặc thiết bị edge siêu nhẹ. Tránh vector gây nhiễu ngữ nghĩa và đội chi phí latency/RAM.
- **Pure Vector:** Tìm kiếm đa phương thức (ảnh/âm thanh), xuyên ngôn ngữ (cross-lingual), hoặc truy vấn trừu tượng không chứa từ khóa chung.

---

## Điều ngạc nhiên nhất khi làm lab này

Độ trễ online feature lookup của Feast trên SQLite chỉ mất ~1.25ms (P99), cực kỳ nhẹ và nhanh; đồng thời thuật toán RRF ($k=60$) đã giúp đưa độ chính xác của các truy vấn hỗn hợp (mixed) chạm mốc 100.0% tuyệt đối.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _None_

