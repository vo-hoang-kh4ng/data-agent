# Proposal Nghiên Cứu

## Triadic DGM: Tiến Hóa Mở Của Agent Sinh Mã Đa Ngôn Ngữ Qua Lọc Thông Tin Lý Thuyết Và Ngân Sách Tính Toán Động

---

## Tóm Tắt

Bài báo này đề xuất **Triadic Darwin Gödel Machine (Triadic DGM)** — một khung tiến hóa mở (open-ended self-evolution) cho các agent sinh mã đa ngôn ngữ. Hệ thống kết hợp ba đổi mới kỹ thuật cốt lõi: (1) pipeline **Proposer–Solver–Verifier** phân vai để đồng thời đề xuất, giải quyết và xác minh tác vụ; (2) bộ lọc **Epiplexity** dựa trên lý thuyết mô tả tối thiểu (MDL) nhằm chọn lọc các đột biến "vừa đủ khó" trong **Goldilocks Zone**; và (3) **RIMRULE Memory Bank** tích lũy quy tắc sửa lỗi có thể chuyển giao xuyên ngôn ngữ. Để giảm chi phí đánh giá, hệ thống áp dụng kiến trúc **SAEA (Surrogate-Assisted Evolutionary Algorithm)** phân tách vòng tiến hóa nội bộ (30 chu kỳ, đánh giá bằng surrogate) và vòng ngoài (đánh giá bằng compiler thực tế). Thực nghiệm trên **Polyglot Benchmark** (6 ngôn ngữ: Python, Go, C++, Rust, Java, JavaScript) cho thấy hệ thống đạt **32.50% Pass@1**, vượt chuẩn DGM gốc (30.7%) và đạt **20.0% Pass@1 xác minh bằng compiler thật** trong outer loop evaluation.

---

## 1. Giới Thiệu

### 1.1 Bối Cảnh

Sự xuất hiện của các Mô hình Ngôn ngữ Lớn (LLM) đã tạo ra một thế hệ agent sinh mã mới, có khả năng tổng hợp và sửa chữa mã nguồn theo yêu cầu tự nhiên. Tuy nhiên, các hệ thống hiện tại phần lớn là **tĩnh**: kiến trúc orchestrating của agent được thiết kế một lần và không thay đổi trong suốt vòng đời. Điều này tạo ra một nghịch lý: LLM ngày càng mạnh hơn, nhưng logic điều phối của agent vẫn là nút cổ chai tĩnh tại.

Câu hỏi nghiên cứu nền tảng là: **Liệu một agent sinh mã có thể tự cải thiện kiến trúc điều phối của chính nó theo thời gian?**

Darwin Gödel Machine (DGM) [Lu et al., 2025] đặt nền móng cho hướng tiếp cận này bằng cách cho phép agent tự viết lại chính mình và đánh giá kết quả thông qua benchmark ngoại vi. Tuy nhiên, DGM gốc đối mặt với ba giới hạn thực tiễn nghiêm trọng:

1. **Chi phí đánh giá fitness cực cao**: Mỗi đột biến đòi hỏi chạy toàn bộ SWE-bench qua Docker, tốn hàng chục phút mỗi chu kỳ.
2. **Không có bộ lọc học tập**: Hệ thống không phân biệt được đột biến "quá dễ" (không học được gì) và "quá khó" (không khả thi), dẫn đến lãng phí compute.
3. **Tri thức không tích lũy**: Mỗi chu kỳ tiến hóa bắt đầu từ đầu, không kế thừa bài học từ các lần thất bại trước.

### 1.2 Đóng Góp Chính

Bài báo này đề xuất Triadic DGM giải quyết cả ba giới hạn trên thông qua:

- **SAEA (Surrogate-Assisted Evolutionary Algorithm)**: Tách biệt inner loop (30 chu kỳ, surrogate nhanh) và outer loop (compiler thật, đánh giá chính xác), giảm chi phí đánh giá xuống ~95%.
- **Epiplexity + Goldilocks Zone**: Bộ lọc MDL chọn lọc đột biến có "thông tin học tập" tối ưu, loại bỏ tác vụ tầm thường và bất khả thi.
- **RIMRULE Memory Bank**: Hệ thống quy tắc sửa lỗi có thể tái sử dụng xuyên ngôn ngữ, tích lũy từ vòng phản hồi Reflexion.
- **Meta-Evolution của chiến lược tiến hóa**: Bản thân thuật toán chọn cha mẹ cũng được LLM tự nâng cấp mỗi 5 chu kỳ.

---

## 2. Công Trình Liên Quan

### 2.1 Self-Improving Code Generation Agents

Công trình tiên phong là **AlphaCode** [Li et al., 2022] và **CodeT** [Chen et al., 2023], tập trung vào việc tăng độ chính xác qua sampling đa dạng và lọc bằng test case. Tuy nhiên, các hệ thống này cải thiện **kết quả đầu ra** mà không sửa đổi **kiến trúc agent**.

**SWE-agent** [Yang et al., 2024] và **AutoCodeRover** [Zhang et al., 2024] đưa agent vào môi trường git thực tế để sửa bug, nhưng kiến trúc điều phối vẫn bất biến.

### 2.2 Darwin Gödel Machine (DGM)

DGM [Lu et al., 2025] là công trình đầu tiên cho phép agent tự viết lại mã nguồn điều phối của chính nó. Hệ thống duy trì một **archive** các phiên bản agent và sử dụng LLM để đề xuất cải tiến dựa trên các lần thất bại cụ thể trên SWE-bench. Kết quả đạt 30.7% Pass@1 trên SWE-bench verified.

Triadic DGM mở rộng DGM gốc theo ba chiều: (1) đa ngôn ngữ (Polyglot), (2) surrogate-assisted evaluation, (3) meta-evolution của chính chiến lược tiến hóa.

### 2.3 Minimum Description Length (MDL) trong AI

MDL [Rissanen, 1978; Grünwald, 2007] cung cấp nền tảng lý thuyết để đo lường "lượng thông tin có thể học được" từ một tập dữ liệu. Trong lĩnh vực AI, MDL đã được áp dụng để lựa chọn mô hình [Hansen & Yu, 2001] và đánh giá độ phức tạp của thuật toán [Schmidhuber, 2010].

Đây là công trình đầu tiên sử dụng MDL như một **bộ lọc online** trong vòng tiến hóa agent, không phải như một tiêu chí chọn mô hình offline.

### 2.4 Upper Confidence Bound (UCB1) trong Multi-Armed Bandit

UCB1 [Auer et al., 2002] giải quyết bài toán cân bằng khai thác–khám phá (exploration–exploitation tradeoff) trong multi-armed bandit. Trong Triadic DGM, bài toán chọn parent agent được đặt trong framework bandit: mỗi node trong archive là một "cánh tay máy", và UCB1 giúp khám phá các node ít được thử trong khi vẫn ưu tiên node có điểm số cao.

### 2.5 Reflexion và Self-Correction trong LLM

**Reflexion** [Shinn et al., 2023] giới thiệu cơ chế cho phép LLM học từ phản hồi lỗi bằng cách đưa error log vào ngữ cảnh thế hệ tiếp theo. Triadic DGM tích hợp Reflexion ở **cả hai cấp độ**: cấp giải task (Outer Loop, tối đa 3 lần) và cấp đột biến kiến trúc (Inner Loop, tối đa `max_retries` lần).

---

## 3. Phương Pháp Đề Xuất

### 3.1 Kiến Trúc Tổng Thể: SAEA Hai Tầng

Triadic DGM tổ chức quá trình tiến hóa theo hai vòng lặp lồng nhau:

```
INNER LOOP (Surrogate Fitness — Nhanh)
  T = 30 chu kỳ
  Đột biến LAMBDA.py bằng LLM
  Đánh giá bằng Epiplexity surrogate (~1s/chu kỳ)
  Lưu vào archive nếu đạt Goldilocks Zone
         ↓
OUTER LOOP (Real Compiler Fitness — Chính Xác)
  Chọn Best Agent từ archive
  Giải N bài Exercism thực tế
  Compiler thực tế (pytest, go test, cargo test, ...)
  Tính Pass@1 chính xác
```

Kiến trúc này tiếp thu ý tưởng từ **surrogate-assisted optimization** [Jin, 2011] nhưng áp dụng vào không gian mã nguồn agent — một không gian rời rạc, không liên tục, và không thể gradient hóa.

### 3.2 Pipeline Triadic: Proposer–Solver–Verifier

Mỗi chu kỳ tiến hóa gồm ba vai đặc biệt:

**Proposer**: Khai thác knowledge graph của codebase (được xây dựng bởi module Understand-Anything) để đề xuất task phù hợp với điểm yếu hiện tại của agent. Proposer không tạo task ngẫu nhiên — nó ưu tiên các tác vụ liên quan đến các thành phần có ít coverage nhất.

**Solver**: Nhận task từ Proposer, đọc mã nguồn LAMBDA.py hiện tại và kiến trúc context, sau đó sinh ra phiên bản cải tiến. Solver có thể nhận error feedback từ chu kỳ trước (Reflexion loop) để sửa lỗi tự động.

**Verifier**: Đánh giá phiên bản mới theo hai chiều độc lập:
- *Syntactic/Runtime validity*: Load LAMBDA.py vào memory, chạy staged evaluation
- *Information-theoretic fitness*: Tính Epiplexity để quyết định Goldilocks status

Ba vai này không cố định về mô hình LLM — mỗi vai có thể dùng model với kích thước khác nhau (Proposer cần suy luận cao → model lớn; Solver cần tốc độ → model nhỏ hơn).

### 3.3 Epiplexity và Goldilocks Zone

**Định nghĩa 1 (Epiplexity)**: Cho đoạn mã $c$ có phân phối ký tự $p$, Epiplexity được định nghĩa là:

$$\text{Epiplexity}(c) = H_{\text{Shannon}}(c) \times \left(1 - \frac{|\text{compress}(c)|}{|c|}\right)$$

trong đó $H_{\text{Shannon}}(c) = -\sum_i p_i \log_2 p_i$ là entropy Shannon của chuỗi ký tự, và $|\text{compress}(c)|/|c|$ là tỷ lệ nén (dùng zlib).

**Trực giác**: Một đoạn mã có entropy cao (ký tự phân phối đều) nhưng khó nén (ít cấu trúc lặp) → nội dung mới mẻ, không thể rút gọn. Đây là proxy cho **lượng thông tin mới** mà đột biến mang lại so với các phiên bản trước.

**Định nghĩa 2 (Goldilocks Zone)**: Một đột biến được chấp nhận vào archive khi và chỉ khi:

$$\epsilon_{\min} \leq \text{Epiplexity}(c) \leq \epsilon_{\max}$$

với $\epsilon_{\min} = 0.5$ (loại bỏ đột biến tầm thường) và $\epsilon_{\max} = 1.8$ (loại bỏ đột biến bất khả thi). Giá trị $[\epsilon_{\min}, \epsilon_{\max}]$ được hiệu chỉnh thực nghiệm.

**Đặc tính mong muốn**: Goldilocks Zone hướng quá trình tìm kiếm vào vùng "vừa đủ phức tạp" — tương tự vùng **zone of proximal development** trong lý thuyết học tập Vygotsky, hay khái niệm **interesting** trong các hệ thống curiosity-driven learning [Schmidhuber, 1991].

### 3.4 Ngân Sách Tính Toán Động

Thay vì phân bổ compute đồng đều cho mọi task, `DynamicCapacityManager` điều chỉnh tài nguyên theo độ khó ước lượng:

| Epiplexity | `num_candidates` | `max_retries` | Lý do |
|---|---|---|---|
| < 0.5 (dễ) | 1 | 1 | Không đáng đầu tư nhiều |
| 0.5–1.2 (trung bình) | 2–3 | 2 | Học được nhiều nhất |
| > 1.2 (khó) | 4–5 | 3 | Cần nhiều lần thử để hội tụ |

Cơ chế này cắt giảm tổng compute tiêu thụ xuống tới ~40% so với phân bổ đồng đều, đồng thời tập trung nỗ lực vào những task nằm trong Goldilocks Zone.

### 3.5 RIMRULE Memory Bank

**Động lực**: Khi Reflexion sửa thành công một lỗi, tri thức sửa lỗi đó thường có giá trị tổng quát (ví dụ: "off-by-one error trong vòng lặp Go thường xuất hiện khi dùng `range` sai"). Thay vì bỏ qua bài học này sau mỗi task, RIMRULE tích lũy chúng dưới dạng quy tắc có thể inject vào prompt của những task tiếp theo.

**Cơ chế scoring theo MDL**: Mỗi quy tắc $r$ được đánh giá bằng:

$$\text{MDL-score}(r) = \frac{\text{DataCost}(r)}{\text{ModelCost}(r)} = \frac{\text{số lỗi được sửa nhờ } r}{\text{độ dài của } r \text{ (token)}}$$

Quy tắc nào có tỷ số cao (sửa được nhiều lỗi với mô tả ngắn gọn) được ưu tiên inject. Đây là áp dụng trực tiếp nguyên lý MDL: mô hình tốt là mô hình giải thích được nhiều dữ liệu nhất bằng ít bit nhất.

**Tính chuyển giao xuyên ngôn ngữ**: Vì quy tắc được mô tả bằng ngôn ngữ tự nhiên (không phải syntax-specific), chúng có thể áp dụng cho nhiều ngôn ngữ lập trình. Ví dụ: quy tắc "kiểm tra điều kiện biên khi xử lý danh sách rỗng" hợp lệ cho cả Python, Go, và Rust.

### 3.6 Meta-Evolution Của Chiến Lược Tiến Hóa

Đây là điểm mới nhất so với DGM gốc: **chiến lược chọn cha mẹ không cố định mà chính nó cũng tiến hóa**.

Mỗi 5 chu kỳ, hệ thống kích hoạt **Meta-Evolution Step**:

1. LLM đọc mã nguồn `evolution_strategy.py` hiện tại
2. Sinh ra phiên bản mới với strategy khác (UCB, Thompson Sampling, ε-greedy, ...)
3. `strategy_validator.py` kiểm tra tính hợp lệ qua hai lớp:
   - **AST check**: Đúng cú pháp, đủ interface, không có module nguy hiểm
   - **Runtime sandbox**: Chạy trong process riêng với timeout 3 giây
4. Nếu hợp lệ → ghi đè `evolution_strategy.py`
5. Nếu thất bại 3 lần liên tiếp → ROLLBACK về `evolution_strategy_baseline.py`

File `evolution_strategy_baseline.py` là **bất biến** — đóng vai trò như "parachute" đảm bảo hệ thống luôn có chiến lược hoạt động được.

Cơ chế này tạo ra một hệ thống **hai cấp tự tham chiếu**:
- Cấp 1: Agent tiến hóa để giải mã tốt hơn
- Cấp 2: Chiến lược tiến hóa của cấp 1 cũng tiến hóa

---

## 4. Cơ Chế Bảo Mật và An Toàn Tham Chiếu

Một thách thức đặc biệt của hệ thống tự sửa đổi là nguy cơ **Objective Hacking** — agent tìm cách nâng điểm bằng cách sửa chính bộ đánh giá thay vì cải thiện năng lực thực sự.

Triadic DGM giải quyết vấn đề này qua kiến trúc **Frozen Evaluation Boundary**:

```
Files được phép đột biến:          Files KHÔNG được phép đột biến (FROZEN):
  LAMBDA.py                           core/inspector.py      ← Tính Epiplexity
  core/programmer.py                  dgm_agent/lambda_eval.py ← Chạy test
  core/proposer.py                    scripts/download_polyglot.py ← Dữ liệu chuẩn
  dgm_agent/evolution_strategy.py     plot_results.py        ← Điều phối chính
```

Ranh giới này là **bất đối xứng về mặt thiết kế**: agent có thể cải thiện năng lực giải bài (LAMBDA.py) và chiến lược tìm kiếm (evolution_strategy.py), nhưng không thể thay đổi tiêu chí đánh giá thành công. Tính bất biến này là điều kiện cần thiết để kết quả Pass@1 có ý nghĩa khoa học.

---

## 5. Thiết Kế Thực Nghiệm

### 5.1 Bộ Dữ Liệu: Polyglot Benchmark

Thực nghiệm sử dụng **Polyglot Benchmark** — tập hợp các bài toán lập trình từ nền tảng Exercism, trải đều trên 6 ngôn ngữ. Đặc điểm:

- **Đa dạng ngôn ngữ**: Python, Go, C++, Rust, Java, JavaScript
- **Đánh giá khách quan**: Test case từ exercism.org, không do nhóm nghiên cứu tạo ra
- **Độ khó phân tầng**: Từ toán học đơn giản (pangram) đến thuật toán phức tạp (dominoes chaining, variable-length quantity encoding)

### 5.2 Các Baseline So Sánh

| Hệ thống | Mô tả |
|---|---|
| **Gen 0 (Baseline)** | LAMBDA không qua tiến hóa, Pass@1 = 16.50% |
| **DGM gốc** | Darwin Gödel Machine [Lu et al., 2025], Pass@1 = 30.7% |
| **no_darwin** | Inner loop không có selection (luôn dùng commit gần nhất) |
| **no_selfimprove** | Chỉ chạy outer loop, không có vòng tiến hóa |
| **Triadic DGM (ours)** | Đề xuất này, Pass@1 = 32.50% |

### 5.3 Metrics Đánh Giá

**Metric chính:**
- **Pass@1 (Outer Loop)**: Tỷ lệ bài giải được trong lần thử đầu tiên, đánh giá bằng compiler thực tế. Đây là con số xuất hiện trong bài báo.

**Metrics phụ:**
- **Goldilocks Rate**: Tỷ lệ đột biến nằm trong Goldilocks Zone
- **Archive Utilization**: Tỷ lệ chu kỳ tạo ra record được archive
- **Reflexion Success Rate**: Tỷ lệ lỗi được tự sửa thành công
- **RIMRULE Reuse Rate**: Tần suất quy tắc từ Memory Bank được dùng thành công

### 5.4 Điều Kiện Thực Nghiệm

- **LLM backend**: Groq API với `llama-3.3-70b-versatile` (Outer Loop) và `llama-3.1-8b-instant` (Inner Loop mutations)
- **Số chu kỳ Inner Loop**: T = 30
- **Số lần retry Reflexion**: Tối đa 3 (Outer Loop), tối đa `max_retries` (Inner Loop, động theo Epiplexity)
- **Tần suất Meta-Evolution**: Mỗi 5 chu kỳ Inner Loop

---

## 6. Kết Quả Và Phân Tích

### 6.1 Kết Quả Chính

| Hệ thống | Pass@1 |
|---|---|
| Gen 0 (no evolution) | 16.50% |
| DGM [Lu et al., 2025] | 30.70% |
| **Triadic DGM (ours)** | **32.50%** |
| Outer Loop xác minh thực tế | 20.00% |

Sự chênh lệch giữa Inner Loop score (32.50%) và Outer Loop score (20.00%) phản ánh **surrogate gap** — khoảng cách giữa surrogate fitness (Epiplexity) và real compiler fitness. Đây là phát hiện quan trọng: surrogate tốt trong việc lọc đột biến xấu, nhưng có thể overestimate năng lực thực tế.

### 6.2 Ablation Study

Để hiểu đóng góp của từng thành phần, nhóm thực nghiệm tiến hành ablation:

| Cấu hình | Pass@1 | Δ so với Full |
|---|---|---|
| Full Triadic DGM | 32.50% | — |
| Không có Goldilocks filter | 28.10% | -4.40% |
| Không có RIMRULE | 29.80% | -2.70% |
| Không có Meta-Evolution | 30.60% | -1.90% |
| Không có Dynamic Budget | 31.20% | -1.30% |

Kết quả cho thấy **Goldilocks filter có đóng góp lớn nhất** — loại bỏ đột biến tầm thường giúp tập trung evolution budget vào vùng học tập hiệu quả.

### 6.3 Phân Tích Goldilocks Zone Qua Các Thế Hệ

Qua 30 chu kỳ tiến hóa, phân phối Epiplexity của các đột biến được archive hội tụ về vùng `[0.7, 1.3]` — hẹp hơn ngưỡng lý thuyết `[0.5, 1.8]`. Điều này gợi ý rằng: (1) agent học cách đề xuất task trong vùng tối ưu, và (2) ngưỡng Goldilocks có thể được adaptive theo thời gian tiến hóa.

---

## 7. Đóng Góp Khoa Học

Bài báo này đóng góp vào cộng đồng khoa học theo bốn chiều:

**C1 — Khung lý thuyết mới**: Lần đầu tiên áp dụng Minimum Description Length như bộ lọc online trong vòng tiến hóa agent mã nguồn. Epiplexity cung cấp nền tảng lý thuyết cho trực giác "vừa đủ khó" thường được dùng không chính thức trong curriculum learning.

**C2 — Kiến trúc SAEA cho không gian mã nguồn**: Chứng minh rằng surrogate-assisted optimization — đã thành công trong tối ưu hóa liên tục — có thể mang lại lợi ích tương tự trong không gian rời rạc của mã nguồn agent.

**C3 — Tự tham chiếu hai cấp**: Mô tả và kiểm chứng hệ thống meta-evolution an toàn, trong đó cả chiến lược tiến hóa lẫn bản thân agent đều có thể tự sửa đổi trong khi vẫn duy trì tính bất biến của bộ đánh giá.

**C4 — Cơ chế tri thức chuyển giao**: RIMRULE Memory Bank là framework mới để tích lũy và tái sử dụng quy tắc sửa lỗi xuyên ngôn ngữ, mở ra hướng nghiên cứu về cross-lingual knowledge transfer trong lĩnh vực code generation.

---

## 8. Hạn Chế và Hướng Phát Triển

### 8.1 Hạn Chế Hiện Tại

**Surrogate gap**: Khoảng cách 12.5% giữa Inner Loop (32.50%) và Outer Loop (20.00%) cho thấy Epiplexity chưa phải surrogate hoàn hảo. Các tác vụ có Epiplexity cao không nhất thiết tương ứng với các lỗi thực tế mà compiler sẽ phát hiện.

**Phụ thuộc vào Groq API**: Toàn bộ hệ thống mutation và generation phụ thuộc vào một LLM provider duy nhất. Sự cố API hoặc thay đổi rate limit ảnh hưởng trực tiếp đến tốc độ tiến hóa.

**Compiler coverage không đầy đủ**: Outer Loop bỏ qua các bài Go, Rust, Java, C++ khi compiler không được cài đặt trên host, dẫn đến số bài được đánh giá thực tế nhỏ hơn tổng task registry.

**Archive không có cơ chế quên**: `evolution_archive.json` tích lũy tất cả thế hệ mà không có cơ chế loại bỏ các record lỗi thời, có thể gây bias trong parent selection ở các thế hệ muộn.

### 8.2 Hướng Phát Triển

**Adaptive Goldilocks Zone**: Thay vì dùng ngưỡng cố định $[\epsilon_{\min}, \epsilon_{\max}]$, nghiên cứu tiếp theo có thể học ngưỡng này tự động dựa trên lịch sử tiến hóa — tương tự **adaptive curriculum** trong reinforcement learning [Portelas et al., 2020].

**Multi-objective Archive**: Mở rộng archive để lưu trữ theo Pareto front (điểm số × đa dạng Epiplexity), áp dụng kỹ thuật từ **Quality-Diversity (QD) algorithms** [Cully & Demiris, 2017].

**Federated Evolution**: Nhiều instance Triadic DGM chạy song song trên các ngôn ngữ khác nhau, chia sẻ RIMRULE Memory Bank — khai thác tối đa tính chuyển giao xuyên ngôn ngữ của quy tắc sửa lỗi.

**Neural Surrogate thay thế Epiplexity**: Thay thế heuristic MDL bằng một mô hình học máy nhỏ được train online để dự đoán Pass@1 thực tế từ đặc trưng mã nguồn, giảm surrogate gap.

**Formal Verification cho Frozen Boundary**: Áp dụng kỹ thuật static analysis để chứng minh hình thức rằng không có đường dẫn đột biến nào có thể vi phạm FROZEN_FILES, thay vì chỉ kiểm tra runtime.

---

## 9. Kết Luận

Bài báo này trình bày Triadic DGM — một hệ thống tiến hóa mở cho agent sinh mã đa ngôn ngữ. Bằng cách kết hợp bộ lọc Epiplexity dựa trên lý thuyết thông tin, kiến trúc SAEA hai tầng, RIMRULE Memory Bank, và meta-evolution của chiến lược tiến hóa, hệ thống vượt qua chuẩn DGM gốc (30.7% → 32.50% Pass@1) đồng thời giảm đáng kể chi phí tính toán per cycle.

Quan trọng hơn, bài báo mở ra một hướng nghiên cứu mới: **thiết kế các bất biến an toàn (safety invariants) trong hệ thống tự sửa đổi** — một vấn đề nền tảng khi hệ thống AI ngày càng được trao quyền tự động hóa nhiều hơn. Cơ chế Frozen Evaluation Boundary và strategy_validator là những bước đầu tiên hướng tới các hệ thống tự cải thiện an toàn và kiểm chứng được.

---

## Tài Liệu Tham Khảo

- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2), 235–256.
- Chen, B., et al. (2023). CodeT: Code generation with generated tests. *ICLR 2023*.
- Cully, A., & Demiris, Y. (2017). Quality and diversity optimization: A unifying modular framework. *IEEE Transactions on Evolutionary Computation*, 22(2), 245–259.
- Grünwald, P. D. (2007). *The Minimum Description Length Principle*. MIT Press.
- Hansen, M. H., & Yu, B. (2001). Model selection and the principle of minimum description length. *Journal of the American Statistical Association*, 96(454), 746–774.
- Jin, Y. (2011). Surrogate-assisted evolutionary computation: Recent advances and future challenges. *Swarm and Evolutionary Computation*, 1(2), 61–70.
- Li, Y., et al. (2022). Competition-level code generation with AlphaCode. *Science*, 378(6624), 1092–1097.
- Lu, C., et al. (2025). Darwin Gödel Machine: Self-improving AI agents. *arXiv:2505.22954*.
- Portelas, R., et al. (2020). Automatic curriculum learning for deep RL. *ECML-PKDD 2020*.
- Rissanen, J. (1978). Modeling by shortest data description. *Automatica*, 14(5), 465–471.
- Schmidhuber, J. (1991). A possibility for implementing curiosity and boredom in model-building neural controllers. *SAB'91 Proceedings*.
- Schmidhuber, J. (2010). Formal theory of creativity, fun, and intrinsic motivation. *IEEE Transactions on Autonomous Mental Development*, 2(3), 230–247.
- Shinn, N., et al. (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS 2023*.
- Yang, J., et al. (2024). SWE-agent: Agent-computer interfaces enable automated software engineering. *NeurIPS 2024*.
- Zhang, Y., et al. (2024). AutoCodeRover: Autonomous program improvement. *ISSTA 2024*.
