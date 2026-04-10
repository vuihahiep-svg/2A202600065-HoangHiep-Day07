# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Hoàng Hiệp - 2A202600065 
**Nhóm:** 14
**Ngày:** 10/04/2026  

***

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (giá trị tiến gần đến 1.0) nghĩa là hai vector có hướng rất giống nhau trong không gian đa chiều, cho thấy hai đoạn văn bản có sự tương đồng rất lớn về mặt ngữ nghĩa, bất kể độ dài ngắn của chúng.

**Ví dụ HIGH similarity:**
* **Sentence A:** "Học viên được phép nghỉ tối đa 04 buổi trong suốt chương trình."
* **Sentence B:** "Quy định chuyên cần cho phép học viên vắng mặt không quá 4 lần."
* **Tại sao tương đồng:** Cả hai câu cùng mô tả một quy tắc quản lý lớp học với cùng giới hạn (4 buổi), dù sử dụng hệ thống từ vựng khác nhau (được phép nghỉ / vắng mặt).

**Ví dụ LOW similarity:**
* **Sentence A:** "Chương trình sử dụng LearnWorlds làm hệ thống Quản lý Học tập."
* **Sentence B:** "Sinh viên sẽ thực tập tại công ty VinSmartFuture hoặc VinSOC."
* **Tại sao khác:** Hai câu thuộc hai khía cạnh hoàn toàn tách biệt của chương trình (công cụ phần mềm vs môi trường thực tập doanh nghiệp), không có điểm chung về ngữ cảnh.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo lường góc giữa hai vector (tập trung vào ý nghĩa), trong khi Euclidean đo lường khoảng cách tuyệt đối (bị ảnh hưởng nặng bởi độ dài văn bản). Một đoạn quy chế dài và một câu tóm tắt ngắn có thể mang cùng ý nghĩa, Euclidean sẽ coi chúng khác xa nhau, nhưng Cosine vẫn nhận diện được sự tương đồng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> * Bước nhảy (step) = 500 - 50 = 450.
> * Công thức: ceil((Tổng_ký_tự - Overlap) / Step)
> * Phép tính: ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23.
> * **Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ **tăng lên thành 25 chunks** (vì bước nhảy giảm xuống còn 400). Chúng ta muốn tăng overlap để đảm bảo các thông tin quan trọng nằm ở ranh giới giữa hai chunk (ví dụ: một quy định nằm vắt ngang giữa hai đoạn) không bị cắt đứt ngữ cảnh, giúp LLM hiểu trọn vẹn ý khi truy xuất.

***

## 2. Document Selection (Nhóm) (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Tài liệu Quy chế, Hướng dẫn và FAQ của Chương trình Đào tạo Nhân tài AI Thực chiến Tập đoàn Vingroup (phối hợp cùng VinUni).

**Tại sao nhóm chọn domain này?**
> Bộ tài liệu này có cấu trúc phân cấp rất rõ ràng (theo từng mục I, II, III...) chứa nhiều luật lệ chặt chẽ về học vụ (thời khóa biểu, quy định chuyên cần, bồi hoàn trợ cấp). Đây là use case thực tế và hoàn hảo để kiểm tra khả năng của RAG trong việc tạo ra một "Tư vấn viên ảo" trả lời thắc mắc chính xác cho học viên.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `01_Thong-tin-chung.txt`  | VinUni Docs | 3,365 | `category: overview` |
| 2 | `02_Cau-truc-dao-tao.txt` | VinUni Docs | ~5,500 | `category: curriculum` |
| 3 | `03_Cong-ty-thuc-tap.txt`| VinUni Docs | ~4,200 | `category: internship` |
| 4 | `08_Quy-trinh-dao-tao.txt`| VinUni Docs | 8,758 | `category: policy` |
| 5 | `FAQ.txt` | VinUni Docs | ~8,500 | `category: faq` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | `policy` | Giúp thu hẹp phạm vi tìm kiếm khi user hỏi về "luật lệ/phạt" thay vì hỏi về "chương trình học" (`curriculum`). |
| `doc_id` | string | `FAQ_01` | Giúp hệ thống dễ dàng xóa (`delete_document`) hoặc update lại vector khi file nội quy có sự thay đổi. |

***

## 3. Chunking Strategy (15 điểm)

### Baseline Analysis

Chạy thử nghiệm nghiệm `ChunkingStrategyComparator().compare()` trên hai tài liệu tiêu biểu với **Target Constraint: 250 characters/chunk**:

**Tài liệu 1: `08_Quy-trinh-dao-tao.txt` (Tổng: 8758 Chars)**
| Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|-------------|------------|-------------------|
| FixedSizeChunker (`fixed_size`) | 44 | 247.9 | Low (Cắt ngang từ/câu) |
| SentenceChunker (`by_sentences`) | 21 | 415.8 | High (Khá dài) |
| RecursiveChunker (`recursive`) | 52 | 166.5 | High |
| RegexChunker (`regex_pattern`) | 31 | 280.7 | Rất Cao (Tối ưu nhất) |

**Tài liệu 2: `01_Thong-tin-chung.txt` (Tổng: 3365 Chars)**
| Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|-------------|------------|-------------------|
| FixedSizeChunker (`fixed_size`) | 17 | 245.0 | Low (Cắt ngang tiêu đề) |
| SentenceChunker (`by_sentences`) | 11 | 303.9 | High |
| RecursiveChunker (`recursive`) | 20 | 166.6 | High |
| RegexChunker (`regex_pattern`) | 6 | 559.3 | High (Dài vượt target) |

### Strategy Của Tôi

**Loại:** `RegexChunker` (Kết hợp fallback sang `RecursiveChunker`)

**Mô tả cách hoạt động:**
> `RegexChunker` sử dụng biểu thức chính quy (Regular Expressions) để tìm các dấu hiệu chia tách nội dung cứng như các đường `====` hoặc `---` đặc trưng trong bộ tài liệu VinUni. Nó chia tài liệu thành các block logic độc lập (ví dụ: một điều luật trọn vẹn) trước khi xét đến giới hạn ký tự.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Phân tích bảng số liệu cho thấy một insight cực kỳ đắt giá: `RegexChunker` phụ thuộc mạnh vào cấu trúc file.
> - Ở **File 08 (Quy chế)**, có rất nhiều điều khoản nhỏ được phân tách bằng `---`, `RegexChunker` hoạt động hoàn hảo với `Avg Length: 280.7` (rất sát target 250) và tạo ra 31 chunks giữ nguyên vẹn 100% ngữ cảnh luật lệ.
> - Ở **File 01 (Giới thiệu)**, do ít dải phân cách hơn, nó gộp thành các khối rất lớn (`Avg Length: 559.3`).
> 
> => **Chiến lược cuối cùng của tôi:** Sử dụng `RegexChunker` để chia các khối cấu trúc lớn, nếu khối nào vượt quá 500 ký tự (như ở File 01), sẽ tiếp tục đưa qua `RecursiveChunker` để chẻ nhỏ. Điều này khắc phục được nhược điểm của cả hai: giữ cấu trúc tổng thể và đảm bảo tính nhỏ gọn cho Vector DB.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi Hiệp | `Regex + Recursive` | 9.5/10 | Giữ vẹn nguyên cấu trúc luật lệ, độ dài tối ưu. | Code phức tạp hơn do phải chạy qua 2 lớp logic. |
| Thuận| `Sentence` | 7.5/10 | Giữ được trọn vẹn câu văn. | File 08 cho thấy chunk quá dài (415.8) vượt xa target. |
| Nghĩa| `FixedSize`| 4.0/10 | Target bám sát nhất (247.9 / 250). | Điểm Coherence thấp nhất do cắt đứt ngữ nghĩa ngẫu nhiên. |

***

## 4. My Approach (Cá nhân) (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`**
> Dùng regex `re.split(r'(?<=[.!?])(?:\s|\n)', text)` với kỹ thuật lookbehind để nhận diện ranh giới câu mà không làm mất dấu chấm câu. Gom các câu lại bằng vòng lặp theo giới hạn `max_sentences_per_chunk`.

**`RegexChunker` & `RecursiveChunker`**
> `RegexChunker` dùng `re.split(r"(?m)^(?:#{1,6}\s|={10,}|-{3,})", text)` để tách các block logic của VinUni. Hàm đệ quy `_split` của `RecursiveChunker` hoạt động như một fallback an toàn, chẻ tiếp bằng `\n\n` hoặc `\n` nếu khối regex vẫn lớn hơn giới hạn.

### EmbeddingStore

**`add_documents` + `search`**
> Khởi tạo record dictionary chứa `id`, `content`, `metadata` và vector embedding. Hàm `search` sử dụng hàm `compute_similarity` tính tích vô hướng (dot product) giữa query vector và toàn bộ vector trong store, sort giảm dần lấy top k.

**`search_with_filter` + `delete_document`**
> Chạy vòng lặp filter siêu dữ liệu trên mảng in-memory trước (pre-filtering) để giảm không gian tìm kiếm, sau đó mới tính similarity. Delete thực hiện bằng list comprehension: giữ lại các record có `id` khác với `doc_id` cần xóa.

### KnowledgeBaseAgent

**`answer`**
> Xây dựng prompt RAG chuẩn: Inject toàn bộ nội dung từ các chunk lấy được vào phần `Context:` của prompt, đồng thời thêm chỉ thị yêu cầu LLM "chỉ trả lời dựa trên context được cung cấp, nếu không có thông tin hãy nói không biết" để chống ảo giác (hallucination).

### Test Results

```text
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                          [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                   [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                            [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                             [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                  [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                  [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                        [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                         [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                       [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                         [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                         [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                    [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                          [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                 [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                     [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                               [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                     [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                         [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                           [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                             [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                   [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                        [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                          [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                              [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                           [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                    [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                   [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                              [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                          [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                     [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                         [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                               [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                         [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                      [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                    [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                   [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                       [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                  [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                           [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                 [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                     [100%]

========================================================= 42 passed in 0.04s ======================================================
```

Số tests pass: 42 / 42

## 5. Similarity Predictions (Cá nhân) (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Thời khóa biểu gồm sáng, chiều và tối." | "Lịch học được chia thành các ca trong ngày." | High | 0.86 | Đúng |
| 2 | "Học viên được nhận chứng chỉ VinUni." | "Học viên bị tước quyền nhận chứng chỉ." | Low | 0.79 | Sai |
| 3 | "Thực chiến tại công ty VinSOC." | "Hệ thống LMS LearnWorlds bị lỗi đăng nhập." | Low | 0.15 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là cặp số 2. Dù hai câu mang ý nghĩa trái ngược hoàn toàn (được nhận vs bị tước), chúng vẫn có điểm similarity rất cao. Điều này minh chứng rằng embedding mô phỏng "chủ đề" (nói về chứng chỉ) rất tốt, nhưng đôi khi kém nhạy bén với từ ngữ phủ định/trạng thái đối lập.

***

## 6. Results (Cá nhân) (10 điểm)

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Chương trình có dạy lại kiến thức nền tảng không? | KHÔNG. Chương trình không bao gồm kiến thức nền tảng cơ bản. |
| 2 | Hệ thống học tập trực tuyến (LMS) tên là gì? | Chương trình sử dụng LearnWorlds làm hệ thống Quản lý Học tập. |
| 3 | Giai đoạn 3 thực chiến kéo dài trong bao lâu? | Giai đoạn 3: Thực chiến tại doanh nghiệp (06 tuần). |
| 4 | Học viên được phép nghỉ tối đa bao nhiêu buổi? | Được phép nghỉ tối đa 04 buổi, không được nghỉ 2 buổi liên tiếp 1 tuần. |
| 5 | Các công ty nào nhận thực tập giai đoạn này? | VinSmartFuture, VinSOC, VinRobotics, VinMotion, VinDynamics. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Có dạy kiến thức nền? | "...KHÔNG. Chương trình không bao gồm kiến thức nền..." | 0.94 | Yes | Không, chương trình tập trung vào thực chiến, không dạy lại kiến thức nền. |
| 2 | Hệ thống LMS là gì? | "...Chương trình sử dụng LearnWorlds làm hệ thống..." | 0.96 | Yes | Hệ thống quản lý học tập (LMS) được sử dụng là LearnWorlds. |
| 3 | Thực chiến bao lâu? | "...GIAI ĐOẠN 3: THỰC CHIẾN TẠI DOANH NGHIỆP (06 TUẦN)..."| 0.91 | Yes | Giai đoạn thực chiến tại doanh nghiệp kéo dài trong 6 tuần. |
| 4 | Được nghỉ mấy buổi? | "...Học viên được phép nghỉ tối đa 04 buổi (sáng/chiều)..." | 0.93 | Yes | Bạn được nghỉ tối đa 4 buổi, và không quá 2 buổi liên tiếp trong 1 tuần. |
| 5 | Thực tập ở đâu? | "...Các công ty: VinSmartFuture, VinSOC, VinRobotics..." | 0.89 | Yes | Các công ty bao gồm VinSmartFuture, VinSOC, VinRobotics, VinMotion và VinDynamics. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

***

## 7. What I Learned (5 điểm: Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Nhìn vào dữ liệu thực tế (thay vì chỉ đoán), mình nhận ra `FixedSizeChunker` tuy có thể ép `Avg Len` về sát nhất với Target 250 (đạt 245.0 và 247.9), nhưng lại phá hủy hoàn toàn Coherence do cắt ngang từ/câu. Điều này cho thấy trong RAG, chất lượng ngữ nghĩa quan trọng hơn việc tối ưu size lưu trữ.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Chưa

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thiết lập một quy trình tiền xử lý văn bản (Data Cleaning) trước khi nhúng (embed) như: Xóa các khoảng trắng thừa `\r\n`, chuẩn hóa font chữ về Unicode để file vector không bị nhiễu do lỗi format của file `.txt` gốc.

***

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |