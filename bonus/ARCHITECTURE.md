# Kiến trúc Bộ nhớ Lai (Hybrid Memory Architecture) cho Trợ lý AI Cá nhân Tiếng Việt

**Tác giả:** Đội ngũ Kỹ sư AI Cá nhân hóa (VinUni AICB Track 2 - Day 19)  
**Phiên bản:** 1.0 (Proof of Concept)  
**Mục tiêu:** Xây dựng hệ thống bộ nhớ hai tầng (Two-Tier Hybrid Memory) kết hợp **Episodic Memory** (Vector Database) và **Stable User Profile & Velocity** (Feature Store) nhằm phục vụ trợ lý AI cá nhân hóa dành riêng cho người dùng Việt Nam.

---

## 1. Sơ đồ Kiến trúc Tổng thể (Architecture Diagram)

Kiến trúc tách bạch rõ ràng giữa hai luồng:
1. **Luồng Ghi nhớ (Ingestion / Remember Flow):** Phân đoạn văn bản, tạo vector nhúng và nạp vào Qdrant với metadata cách ly người dùng.
2. **Luồng Hồi tưởng & Ghép ngữ cảnh (Recall & Context Assembly Flow):** Kết hợp đồng thời thuộc tính tĩnh, hành vi tức thời từ Feast Online Store và các đoạn hồi ức liên quan từ Qdrant để tổng hợp thành Prompt hoàn chỉnh cho LLM.

```
                          KIẾN TRÚC HYBRID MEMORY AGENT
                          
     [Người dùng / Hội thoại / Tài liệu mới]
                       │
                       ├────────────────────────────────────────────────┐
                       │                                                │
             (1) Luồng Remember                               (2) Luồng Recall
                       ▼                                                ▼
         ┌───────────────────────────┐                    ┌───────────────────────────┐
         │     Text Chunking         │                    │   Câu hỏi truy vấn        │
         │  (Paragraph/Sentence Split│                    │  "Recommend đọc gì tiếp"  │
         └─────────────┬─────────────┘                    └─────────────┬─────────────┘
                       ▼                                                │
         ┌───────────────────────────┐                                  │
         │     FastEmbed ONNX        │                                  │
         │  (bge-small-en-v1.5 384d) │                                  │
         └─────────────┬─────────────┘                                  │
                       ▼                                                │
         ┌───────────────────────────┐                                  │
         │  Qdrant In-Memory Vector  │ ◄── [Vector Search + Filter] ────┤
         │   (Episodic Memory Store) │     (tenant: user_id = u_001)    │
         │   Payload: user_id, text  │                                  │
         └─────────────┬─────────────┘                                  │
                       │ Top-K Chunks                                   │
                       └────────────────────────┐                       │
                                                ▼                       │
                                   ┌─────────────────────────┐          │
                                   │  Context Assembly Engine│ ◄────────┘
                                   │  (HybridMemoryAgent)    │
                                   └────────────▲────────────┘
                                                │ Features (Latency < 2ms)
                                   ┌────────────┴────────────┐
                                   │   Feast Online Store    │
                                   │     (SQLite / Redis)    │
                                   ├─────────────────────────┤
                                   │ • user_profile_features │
                                   │ • query_velocity_feat   │
                                   └─────────────────────────┘
                                                │
                                                ▼
                                   ┌─────────────────────────┐
                                   │ Assembled Prompt Context│
                                   │  (Gửi tới Generator LLM)│
                                   └─────────────────────────┘
```

---

## 2. Ba Quyết định Kiến trúc & Đánh đổi Kỹ thuật (Architecture Tradeoffs)

### Quyết định 1: Chiến lược phân đoạn bộ nhớ (Chunking Strategy)
* **Phương án lựa chọn:** Phân đoạn theo ranh giới đoạn văn và câu tự nhiên với độ dài tối đa 300 ký tự (Sentence-boundary Semantic Chunking).
* **Phương án so sánh:** 
  - *Option A: Lưu nguyên cuộc hội thoại / tài liệu dài (> 1000 tokens).*
  - *Option B: Chia cứng theo số token cố định (Fixed-size 128 tokens, cắt ngang từ).*
* **Đánh đổi & Lý do lựa chọn (Tradeoff Analysis):**
  - Lưu nguyên văn bản (Option A) làm tăng độ trễ embedding, dễ làm loãng điểm tương đồng Cosine (do vector bị trung bình hóa nhiều ý), và ngốn sạch Context Window của mô hình LLM hạ nguồn.
  - Cắt cứng theo số token (Option B) làm đứt gãy cấu trúc ngữ pháp và làm biến dạng từ ghép tiếng Việt (ví dụ từ "tự động" bị cắt làm đôi ở ranh giới chunk).
  - Phân đoạn theo ranh giới câu (~300 ký tự) là điểm cân bằng lý tưởng: đảm bảo mỗi đoạn hồi ức (memory chunk) chứa trọn vẹn một ý niệm độc lập, tối ưu hóa điểm số Dense Retrieval và dễ dàng đóng gói vào Top-K context.

### Quyết định 2: Lược đồ đặc trưng người dùng (Feature Schema: Tabular vs Latent Embedding)
* **Phương án lựa chọn:** Lược đồ thuộc tính dạng bảng tường minh (Explicit Tabular Features) trên Feast gồm 2 nhóm:
  1. *Stable Profile:* `preferred_language` (String), `reading_speed_wpm` (Int64), `topic_affinity` (String).
  2. *Activity Velocity:* `queries_last_hour` (Int64), `distinct_topics_24h` (Int64).
* **Phương án so sánh:** Biểu diễn người dùng bằng một vector ẩn 256 chiều (User Latent Embedding Vector) học từ lịch sử duyệt bài.
* **Đánh đổi & Lý do lựa chọn:**
  - Latent Embedding có thể nắm bắt các sở thích ngầm phức tạp nhưng là một "hộp đen": khó giải thích (uninterpretable), không thể can thiệp bằng luật nghiệp vụ, và tốn kém tài nguyên tính toán định kỳ.
  - Tabular Features trên Feast có thể kiểm toán được (auditable), dễ dàng cập nhật qua luồng streaming, và cho phép ghép trực tiếp vào câu lệnh System Prompt bằng ngôn ngữ tự nhiên: *"Người dùng đọc với tốc độ 187 wpm, thích chủ đề Cloud, ưu tiên tiếng Việt"*. Điều này giúp LLM điều chỉnh văn phong, độ dài câu trả lời chính xác theo nhu cầu thực tế.

### Quyết định 3: Chiến lược độ tươi dữ liệu (Data Freshness Strategy)
* **Phương án lựa chọn:** Kiến trúc cập nhật hai tốc độ (Dual-Speed Refresh Pipeline):
  - *Tốc độ chậm (Batch Materialization - Chu kỳ ngày):* Cập nhật `user_profile_features` (TTL = 30 ngày). Thuộc tính sở thích đọc và ngôn ngữ mang tính ổn định cao, không cần cập nhật theo từng giây.
  - *Tốc độ nhanh (Streaming / Near Real-time Push - Chu kỳ phút/giờ):* Cập nhật `query_velocity_features` (TTL = 1 giờ). Phản ánh chính xác cường độ hoạt động tức thời của người dùng trong phiên làm việc hiện tại.
* **Đánh đổi & Lý do lựa chọn:**
  - Nếu áp dụng real-time streaming cho toàn bộ thuộc tính, chi phí vận hành Redis/Kafka sẽ đội lên gấp nhiều lần mà không mang lại giá trị gia tăng rõ rệt cho các thuộc tính tĩnh.
  - Ngược lại, nếu chạy batch cho cả `queries_last_hour`, hệ thống sẽ hoàn toàn "mù" trước các đột biến hành vi (như người dùng đang spam tìm kiếm hoặc có dấu hiệu mệt mỏi vào ban đêm).

---

### Quyết định Kiến trúc Bị Bác Bỏ (Rejected Alternative)

* **Phương án bị loại bỏ:** **Lưu trữ toàn bộ Episodic Memory trực tiếp bên trong Feast dưới dạng Embedding Feature View.**
* **Lý do kỹ thuật cụ thể:**
  1. *Khác biệt về chu kỳ cập nhật và khối lượng dữ liệu:* Feature Store (Feast) được thiết kế tối ưu cho các bảng khóa-giá trị (Key-Value) với schema cố định gắn theo Entity ID (`user_id`). Trong khi đó, hồi ức hội thoại (Episodic Memory) là dòng dữ liệu văn bản phi cấu trúc, phát sinh liên tục và không giới hạn độ dài.
  2. *Khả năng tìm kiếm tương đồng (ANN Search):* Các Online Store của Feast (như SQLite, DynamoDB, Redis thuần) không hỗ trợ lập chỉ mục đồ thị Hierarchical Navigable Small World (HNSW) mạnh mẽ và các phép lọc đa điều kiện linh hoạt như một Vector Database chuyên dụng (Qdrant).
  3. Việc tách riêng Qdrant cho Episodic Memory và Feast cho User Features tuân thủ triệt để nguyên lý phân tách trách nhiệm (Separation of Concerns), giúp tối ưu hóa hiệu năng độc lập cho từng thành phần.

---

## 3. Các Yếu tố Đặc thù Ngữ cảnh Tiếng Việt (Vietnamese-Context Considerations)

Hệ thống được thiết kế có chủ đích để giải quyết 4 đặc thù của người dùng Việt Nam:

1. **Hiện tượng pha trộn ngôn ngữ (Code-Switching):**
   - Trong lĩnh vực công nghệ và công việc hàng ngày, người dùng Việt Nam thường xuyên pha trộn thuật ngữ tiếng Anh: *"deploy pod lên cluster"*, *"check lại pipeline CI/CD"*, *"cấu hình autoscaler theo traffic"*.
   - Việc kết hợp BM25 và Vector Search trong bộ nhớ hồi ức giúp bắt trọn vẹn cả từ khóa chuyên ngành verbatim tiếng Anh lẫn diễn đạt ngữ nghĩa tiếng Việt.
2. **Xử lý lỗi gõ dấu Telex và phương ngữ:**
   - Người dùng thường gặp lỗi gõ dính phím Telex hoặc đặt dấu thanh khác nhau (ví dụ: `hoà` vs `hòa`). Vector Embedding giúp bỏ qua sai lệch hình thái từ ở cấp ký tự, duy trì điểm số tương đồng ngữ nghĩa cao.
3. **Đặc thù từ vựng đơn lập (Word Segmentation):**
   - Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không trùng với khoảng trắng (ví dụ: "điện toán đám mây" gồm 4 âm tiết nhưng là 1 khái niệm từ vựng). Khi xây dựng chỉ mục BM25 cho episodic memory, việc xem xét tích hợp thư viện tách từ như `pyvi` hoặc `underthesea` là cần thiết cho production lớn, trong khi ở bản POC, mô hình embedding đa ngữ đã bù đắp được khoảng trống này.
4. **Tuân thủ Nghị định 13/2023/NĐ-CP về Bảo vệ Dữ liệu Cá nhân (PDPD):**
   - Trợ lý cá nhân lưu trữ các đoạn chat nhạy cảm của người dùng. Kiến trúc cài đặt bộ lọc bắt buộc `user_id` ở tầng truy vấn Qdrant (`query_filter=user_filter`), bảo đảm tính cách ly tuyệt đối (Multi-tenant Privacy Isolation), ngăn chặn triệt để nguy cơ người dùng này vô tình nhìn thấy hồi ức của người dùng khác.

---

## 4. Giới hạn của Phiên bản POC & Hướng Phát triển Tiếp theo

* **Cơ chế suy giảm hồi ức (Memory Decay / Forgetting):** Phiên bản hiện tại lưu trữ toàn bộ ký ức vĩnh viễn. Trong tương lai, cần bổ sung hàm tính trọng số thời gian $e^{-\lambda \Delta t}$ để giảm dần mức độ ưu tiên của các ký ức cũ.
* **Tổng hợp ký ức định kỳ (Memory Consolidation):** Khi số lượng mảnh hồi ức vượt quá 10.000 chunks, cần có một tiến trình chạy ngầm gọi LLM để tóm tắt các hội thoại tương đồng thành các bài học cốt lõi (Core Beliefs).
* **Mã hóa dữ liệu tại chỗ (Encryption at Rest):** Cần tích hợp khóa mã hóa riêng biệt cho từng `user_id` đối với cả cơ sở dữ liệu SQLite của Feast và vector storage của Qdrant.
