# Vì sao chưa đạt số của bài báo SLSRec (ML-1M)

Tài liệu này đọc lại toàn bộ kết quả đã chạy trong `experiments/` và liệt kê các nguyên
nhân có thể khiến bản tái hiện không đạt Table II/III, xếp theo mức độ bằng chứng. Bảng
số liệu sinh bằng:

```bash
python scripts/run_variants.py --collect_only --include_legacy --prefix existing --variants lru recblr bdlru wo_sam wo_bdlru wo_p wo_u wo_pu slsrec slsrec_exact
```

(kết quả ghi ra `experiments/compare/existing_ml-1m.md`).

## 1. Hiện trạng

| biến thể | seeds | test R@20 | paper R@20 | lệch so với paper | Δ so với SLSRec (của mình) | Δ so với SLSRec (paper) |
|---|---|---|---|---|---|---|
| LRURec | 42,43,44 | 0.4322 ± 0.0060 | 0.4331 | **−0.23%** | −4.30% | −7.32% |
| RecBLR (`model/recblr.py`) | 42 | 0.4529 | 0.4498 | **+0.69%** | +0.29% | −3.75% |
| w/o U | 42 | 0.4531 | 0.4599 | −1.47% | +0.34% | −1.59% |
| w/o P&U | 42,43,44 | 0.4529 ± 0.0020 | 0.4485 | **+0.98%** | **+0.30%** | **−4.02%** |
| SLSRec (approx, eq. 24) | 42 | 0.4516 | 0.4673 | **−3.37%** | 0 | 0 |
| SLSRec (exact, eq. 19/21) | 42,43,44 | 0.4524 ± 0.0023 | – | – | +0.17% | – |

Chẩn đoán huấn luyện (epoch cuối, `L_per-L_rec` là phần loss tăng thêm do nhiễu,
`α·L_unc/L_rec` là trọng số thêm mà U đặt lên L_rec):

| biến thể | best epoch | val R@20 | L_per − L_rec | α·L_unc/L_rec |
|---|---|---|---|---|
| w/o U | 250 | 0.4787 | +0.0058 | – |
| w/o P&U | 228 | 0.4801 | – | – |
| SLSRec approx | 150 | 0.4814 | +0.0071 | **0.0001** |
| SLSRec exact | 137 | 0.4814 | +0.0071 | 0.287 |

Ba kết luận đọc thẳng từ bảng:

1. **Harness, dữ liệu và cách đánh giá không phải nguyên nhân.** LRURec chạy trong cùng
   harness cho 0.4322 so với 0.4331 của paper (−0.23%, trong độ nhiễu seed).
2. **Backbone SAM + BD-LRU không yếu — còn mạnh hơn paper.** w/o P&U đạt 0.4529, *cao
   hơn* 0.4485 của paper và cao hơn cả RecBLR trong paper (0.4498).
3. **Toàn bộ khoảng cách nằm ở P và U.** Trong paper, thêm P&U vào w/o P&U tăng +4.2%
   R@20 (0.4485 → 0.4673). Ở đây thêm P&U **không tăng gì** (0.4529 → 0.4516–0.4524,
   nằm trong độ nhiễu ±0.002). SLSRec của mình vì thế chỉ ngang w/o P&U.

## 2. Các nguyên nhân có thể, theo mức độ bằng chứng

### 2.1 Uncertainty (U) với loss xấp xỉ eq. (22)/(24) tự tắt sau vài epoch — *đã đo được*

Với β = 0, gradient duy nhất lên `Log σ²` là `−α·e^{−Log σ²}·L_Rec < 0`, nên σ² tăng mãi
tới kẹp `|Log σ²| ≤ 8`. Đo được: `L_unc` giảm 4.65 → 0.71 sau epoch 2 và → 0.0019 ở
epoch 100, `α·L_unc/L_rec = 0.0001` ở cuối, tức `L_total ≈ L_rec + L_per`: **U chỉ có
hiệu lực trong ~5 epoch đầu**, sau đó bằng 0. Ảnh hưởng còn lại chỉ là làm lệch quỹ đạo
ban đầu: đổi vị trí dropout trong nhánh U (eq. 20 in ra so với Fig. 2) làm R@20 đổi
0.4481 ↔ 0.4516 — cỡ độ nhiễu seed, không phải một hiệu ứng điều chuẩn.

Với loss chính xác eq. (19)/(21), `L_unc/L_rec ≈ 0.955` suốt quá trình (1/σ² ≈ hằng số
hơi lớn hơn 1): `L_unc` gần như chỉ là một bản sao co giãn của `L_rec`, mà AdamW gần như
bất biến với việc nhân hệ số cho loss → cũng không có tác dụng (exact 0.4524 ≈ w/o U).

Paper cho w/o U giảm 1.6%. Nghĩa là bản của tác giả *phải* có một cơ chế giữ σ² không
chạy ra vô cực (β > 0, không kẹp, weight decay mạnh trên `Mc`, σ² chuẩn hoá theo batch,
hoặc một công thức khác với cái được in). Bài báo không mô tả cơ chế đó.

### 2.2 Perturbation (P) gần như không tác động — *đã đo được*

`L_per − L_rec` chỉ +0.006–0.007 nats ở cuối huấn luyện: thay 10% vị trí đầu vào BD-LRU
bằng U(0,1) gần như không làm mô hình tệ đi, nên `L_per` chỉ là bản sao của `L_rec` với
mask dropout khác (C θ=0.1: 0.4531 so với θ=0: 0.4500 so với B: 0.4518 — đều trong nhiễu).
Các test `tests/test_perturbation_is_inert.py` (`pytest -m claim`) đo cùng hiện tượng trên
dữ liệu tổng hợp. ~~Lý do cấu trúc: mạng đi vòng qua nhánh G và residual~~ — **sai**, đo trên
checkpoint (mục 4): bỏ H làm CE tăng +2.1 nats, mạng phụ thuộc mạnh vào H. Cơ chế thật là
cổng r_t/i_t của BD-LRU học cách *nhận ra vector U(0,1) và bỏ qua bước đó* — xem mục 4.1.

Khả năng tác giả đặt nhiễu ở chỗ khác — "randomly replacing an item representation in
each layer" có thể hiểu là thay *đầu vào của cả tầng* X⁽ˡ⁻¹⁾ (ảnh hưởng cả SAM lẫn
residual), hoặc thay bằng embedding của một item ngẫu nhiên thay vì vector U(0,1). Bằng
chứng gián tiếp: paper có biến thể *w/o BD-LRU* vẫn còn P, nhưng ở cách hiểu hiện tại P
nằm *bên trong* BD-LRU nên w/o BD-LRU thì P biến mất hoàn toàn.

### 2.3 Mô hình fused (w/o P&U) của mình không overfit trên ML-1M

Paper lập luận P&U chữa overfitting của mô hình fused (Fig. 3: val tụt sau ~40 epoch).
Ở đây w/o P&U có val R@10 tăng đều tới epoch ~225 rồi đi ngang, khoảng cách train/val CE
vừa phải (4.99 so với 5.44). Không có overfitting thì một bộ điều chuẩn chống overfitting
không có gì để sửa. Việc w/o P&U của mình (0.4529) cao hơn của paper (0.4485) gợi ý bản
fused của tác giả khác chi tiết (và overfit hơn) — ví dụ không che padding, attention
nhiều head / có dropout, PFFN khác, không có Conv1-D1, v.v. (Fig. 3 lại là Beauty và
Sports, không phải ML-1M.)

### 2.4 Các chi tiết kiến trúc không được bài báo chốt (ảnh hưởng nhỏ hơn, chưa đo riêng)

| chi tiết | bản hiện tại | phương án khác | flag |
|---|---|---|---|
| weight decay trên λ | có decay (LRURec) → trí nhớ a_t co lại dần | RecBLR: wd = 0 | `--exclude_lambda_from_decay=true` |
| tham số hoá a_t | eq. (11), c = 8 | RecBLR: exp(−softplus(Λ)·r) | dùng `recblr` để so |
| activation sau Conv1-D1 | không (đúng eq. 3) | SiLU như RecBLR | `--sam_conv_activation=silu` |
| PFFN | GEGLU eq. (13), không dropout trong | RecBLR: SiLU-FFN có dropout trong | – |
| positional embedding | có (eq. 1) | RecBLR: không | – |
| item bias trong eq. (15) | không | LRURec: có | `--use_item_bias=true` (đã thử: 0.4490, kém hơn) |
| dropout | 0.2 (Fig. 6) | lưới {0, .2, .4, .6, .8} | `--slsrec_dropout` (0.4: 0.4467, kém hơn) |
| kẹp Log σ² | ±8 | paper không kẹp | `--uncertainty_logvar_clamp=0` |

### 2.5 Cách báo cáo kết quả của bài báo (không kiểm chứng được)

- Cột RecBLR của SLSRec (0.44979) **không** phải số trong bài RecBLR (bài gốc: HR@20 =
  0.4336, chạy RecBole) → tác giả SLSRec đã tự chạy lại RecBLR trong harness LRURec.
  Đó chính là lý do cần chạy `recblr` ở đây: nếu RecBLR của mình cũng ra ≈ 0.450 thì
  harness tương đương của tác giả, và khoảng cách còn lại chắc chắn nằm ở phần riêng của
  SLSRec. **Đã chạy (2026-10-08): 0.4529, +0.7% so với 0.4498 của paper** → harness tương
  đương; khoảng cách nằm ở P/U (mục 4).
- Các bảng độ nhạy (Fig. 6/7, Table V, VI) quét **từng tham số một** trên SLSRec đầy đủ,
  các tham số khác giữ ở mặc định (số của Table II xuất hiện đúng ở điểm mặc định của mọi
  bảng), tức khoảng 5 + 5 + 7 ≈ 15 cấu hình cho mỗi dataset, không phải lưới 175. Trên
  ML-1M, mặc định trùng đúng argmax của cả ba bảng; nếu đó là max của ~15 lần chạy có
  σ ≈ 0.002, kỳ vọng thiên lệch chọn lựa khoảng +0.003–0.004 R@20 — khoảng 1/4 khoảng
  cách 0.016. (Ở dataset khác Table II *không* lấy max: Steam α = 1 cho 0.20844 nhưng
  Table II báo 0.20697 = cột α = 0.3; Sports, Video tương tự.)
- Không nêu số seed, không có độ lệch chuẩn. Ở đây cùng cấu hình + cùng seed + cùng GPU
  thì chạy lại trùng tới chữ số cuối (`D_paper`, `D_papereq`, `D_papereqfix` đều 0.44805),
  nhưng độ lệch chuẩn giữa các seed là 0.002 (w/o P&U) đến 0.006 (LRURec), nên mọi chênh
  lệch đơn-seed dưới ~0.4–1% là nhiễu.

## 3. Thử nghiệm đề xuất (theo thứ tự)

1. ~~**RecBLR gốc trong harness**~~ — **xong**, seed 42: 0.4529 (paper 0.4498). Khớp →
   khoảng cách nằm ở P/U. Thêm seed 43, 44 nếu cần độ lệch chuẩn:
   `python scripts/run_variants.py --variants recblr --seeds 42 43 44`
2. **Toàn bộ Table III trên cùng seed** — so *mẫu hình* giảm của ablation với paper:
   `python scripts/run_variants.py --seeds 42` (recblr, w/o SAM, w/o BD-LRU, w/o P, w/o U,
   w/o P&U, SLSRec). Cột "R@20 vs slsrec" đặt cạnh cột "paper" của Table III.
3. **Các cách hiểu khác của P và U** (đã có flag, mặc định giữ nguyên hành vi cũ):

   ```bash
   python scripts/run_variants.py --variants @p   # P: tắt U, so với wo_pu (không P) và wo_u (P nguyên văn)
   python scripts/run_variants.py --variants @u   # U: tắt P, so với wo_pu (không U) và wo_p (U nguyên văn)
   ```

   | biến thể | flag | giả thuyết kiểm tra |
   |---|---|---|
   | `p_clean_gates` | `--perturbation_gates=clean` | cổng r_t, i_t tính từ x_t sạch, chỉ số hạng đầu vào của (12) bị thay — mạng không thể đóng cổng trước nhiễu (mục 4.1) |
   | `p_bdlru_conv` | `--perturbation_site=bdlru_conv` | nhiễu trước Conv1-D2 (conv làm mượt nhiễu) |
   | `p_layer` | `--perturbation_site=layer` | "thay biểu diễn item ở mỗi tầng" = thay X⁽ˡ⁻¹⁾, chạm cả nhánh cổng và residual — mạng không đi vòng được |
   | `p_embedding` | `--perturbation_site=embedding` | thay embedding X⁽⁰⁾ một lần |
   | `p_shuffle` | `--perturbation_noise=shuffle` | U(0,1) lệch phân phối nên dễ bị nhận ra; dùng vector thật của chuỗi khác |
   | `p_layer_shuffle` | cả hai | nhiễu mạnh nhất và khó nhận ra nhất |
   | `p_element` | `--perturbation_granularity=element` | thay từng thành phần thay vì cả vector |
   | `u_exact` | `--uncertainty_loss=exact` | σ² là nhiệt độ softmax (eq. 19/21), có điểm tối ưu hữu hạn |
   | `u_beta` | `--uncertainty_beta=1` | giữ số hạng β·Log σ (Kendall et al.) — σ² = 2L/β, không chạy ra vô cực |
   | `u_sequence` | `--uncertainty_granularity=sequence` | σ_u một giá trị cho mỗi chuỗi, từ "last row of Xc_u" như bài viết |
   | `u_sequence_beta` | sequence + β = 1 | như trên, không suy biến |
   | `u_global_beta` | `--uncertainty_granularity=global --uncertainty_beta=1` | một σ chung ([45]) — đối chứng năng lực |
   | `u_norm` | `--uncertainty_weight_norm=batch` | 1/σ² chuẩn hoá trung bình 1: U chỉ dồn trọng số, không tự tắt (đối chứng) |

   Cột `L_per-L_rec` cho biết cách đặt nhiễu mới có thực sự làm mô hình khó hơn không;
   `a*L_unc/L_rec` cho biết U còn sống ở cuối huấn luyện không. Cách hiểu nào thắng
   `wo_pu` vượt ngưỡng nhiễu (~1%) thì ghép lại vào SLSRec đầy đủ, ví dụ:
   `python scripts/run_variants.py --variants slsrec --prefix alt -- --perturbation_site=layer --uncertainty_beta=1`.

Chạy thử nhanh toàn bộ pipeline (10 epoch mỗi biến thể, ~3–5 phút/biến thể trên RTX 4050):
`python scripts/run_variants.py --quick`.

## 4. Chẩn đoán cơ chế trên checkpoint (2026-10-08)

Đo bằng `scripts/diagnose_pu.py` trên checkpoint tốt nhất của từng run, với cửa sổ train
ML-1M thật và chính cấu hình của run:

```bash
python scripts/diagnose_pu.py experiments/slsrec/C_perturbation_ml-1m_seed42 experiments/slsrec/B_sam_gate_ml-1m_seed42 --force_p --item_noise 0 0.1 0.2
```

Không tìm thấy lỗi code lệch khỏi công thức: bản mặc định đã được so bit-for-bit với
commit trước, và test kiểm từng số hạng. Cả P lẫn U đều trơ **vì cách hiểu nguyên văn của
bài báo tự vô hiệu hoá chính nó**, đo được như dưới đây.

### 4.1 P: nhiễu ở đầu vào BD-LRU không lọt được vào mô hình

Checkpoint đã hội tụ, θ = 0.1, tầng 1:

| run | train với P? | ΔCE khi bật nhiễu | r_t sạch → ở bước nhiễu | i_t sạch → nhiễu | cos(∇L_rec, ∇L_per) | cos khi chỉ đổi dropout |
|---|---|---|---|---|---|---|
| w/o P&U (B) | không | **+0.125** | 0.515 → 0.460 | 0.351 → 0.266 | **0.27** | 0.72 |
| C, θ = 0 (2 lượt, không nhiễu) | không | +0.074 | 0.621 → 0.586 | 0.464 → 0.321 | 0.23 | 0.79 |
| w/o U (C) | có | **+0.0085** | 0.495 → **0.167** | 0.378 → 0.241 | **0.67** | 0.68 |
| SLSRec approx | có | +0.006 | 0.615 → 0.351 | 0.506 → 0.357 | 0.70 | 0.72 |
| SLSRec exact | có | +0.007 | 0.546 → 0.245 | 0.449 → 0.318 | 0.68 | 0.71 |

Bỏ hẳn H (zero đầu ra BD-LRU) làm CE tăng +2.1–2.3 nats ở mọi run, tức mạng **có** phụ
thuộc mạnh vào nhánh H. Vấn đề không phải nhánh bị bỏ rơi, mà là nhiễu không vào được:

1. **Thông tin của bước bị thay vẫn tới đầu ra nguyên vẹn** qua residual và nhánh cổng
   SAM/Conv1-D1 (cả hai không bị nhiễu), còn trạng thái BD-LRU là trung bình mũ qua nhiều
   bước (a_t ∈ [0.9, 0.999], số hạng đầu vào nhân √(1−a_t²) ≤ 0.44), nên một bước hỏng bị
   pha loãng. Ngay từ epoch 1–20, `L_per − L_rec` lúc train chỉ ≤ 0.005.
2. **Cổng học cách nhận ra U(0,1) và bỏ qua bước đó.** Vector U(0,1) toàn dương, mean 0.5,
   trong khi x_t thật có std ≈ 0.7 và chỉ ~45% thành phần nằm trong [0,1]. Vì eq. (17) thay
   x_t trong *cả* (9)–(12), cổng r_t, i_t nhìn thấy nhiễu, và gradient của chính L_per dạy
   chúng đóng lại (r_t: 0.50 → 0.17, a_t → 1, bước bị bỏ qua). Mô hình chưa từng train với P
   không làm vậy (0.52 → 0.46).
3. Hệ quả: gradient của L_per khác ∇L_rec **đúng bằng** một lần rút dropout khác (0.67 so
   với 0.68), tức `L_total ≈ 2·L_rec` — AdamW bất biến với phép nhân loss, nên P không
   điều chuẩn gì. P đạt mục tiêu cục bộ (bất biến với *chính nhiễu của nó*) nhưng sự bất
   biến đó rẻ và vô dụng:

Kiểm tra thẳng tuyên bố Fig. 4 (thay ρ item đầu vào validation bằng item ngẫu nhiên),
mức giảm R@20:

| run | ρ = 0.1 | ρ = 0.2 |
|---|---|---|
| RecBLR | −12.6% | −27.8% |
| w/o P&U (seed 42 / 43) | −9.0% / −9.7% | −22.0% / −21.9% |
| w/o U (P nguyên văn) | −8.4% | −21.1% |
| C θ = 0 | −9.4% | −21.3% |
| SLSRec approx / exact (seed 42) / exact (seed 43) | −9.3% / −9.6% / −8.7% | −21.6% / −22.1% / −21.9% |

P không làm mô hình bền hơn với item nhiễu thật (chênh ≤ 0.6 điểm, bằng độ lệch giữa hai
seed của B). Mô hình gộp SAM bền hơn RecBLR — do kiến trúc, không phải do P.

**Kiểm chứng cách sửa** (20 epoch, seed 42, tắt U, `--prefix mech20`):

| biến thể | L_per − L_rec lúc train (ep 20) | ΔCE θ = 0.1 | KL vị trí cuối | cos ∇L_per / cos dropout | item-noise ρ = 0.1 / 0.2 |
|---|---|---|---|---|---|
| `wo_u` (nguyên văn) | +0.0025 | +0.0005 | 0.003 | 0.82 / 0.83 | −7.6% / −19.5% |
| `p_clean_gates` | +0.0038 | +0.0024 | 0.004 | 0.81 / 0.81 | −7.0% / −19.7% |
| `p_shuffle` | +0.0034 | −0.0005 | 0.006 | **0.69** / 0.79 | −7.2% / −20.2% |
| `p_layer` | **+0.1145** (vẫn tăng) | **+0.112** | **0.140** | **0.75** / 0.81 | **−6.7% / −18.8%** |

- Chặn đường thoát qua cổng (`p_clean_gates`) hay làm nhiễu khó nhận ra (`p_shuffle`)
  **không đủ**: chừng nào nhiễu chỉ đi vào đầu vào BD-LRU, nguyên nhân (1) vẫn còn.
- Thay *biểu diễn item của cả tầng* X⁽ˡ⁻¹⁾ (`p_layer`, đúng chữ "randomly replacing an
  item representation in each layer") biến L_per thành một mục tiêu thật: chi phí lớn gấp
  ~45 lần lúc train, gradient khác L_rec rõ rệt, và là biến thể bền nhất với item nhiễu.
  Cách hiểu này còn khớp một chi tiết của paper mà cách hiểu nguyên văn không khớp: biến
  thể *w/o BD-LRU* vẫn có P.
- Đây mới là 20 epoch, một seed: nó chứng minh **cơ chế** (P còn sống), chưa chứng minh
  P giúp test metric. Cần chạy đủ (mục 4.4).

### 4.2 U: dạng xấp xỉ eq. (24) tự tắt; dạng chính xác sống nhưng không có lợi

| | approx eq. (24), β = 0 (`D_paperfig`) | exact eq. (19)/(21) (`D_exact`, seed 42 / 43) |
|---|---|---|
| Log σ² p05 / p50 / p95 | 8 / 8 / 8 — **99.6% nằm ở kẹp** | −0.92 / −0.12 / 0.66 |
| α/σ² trung bình (CV) | **0.0001** (0.21) | 0.38 (0.50) |
| Spearman(α/σ², CE) | +0.02 | **−0.70** / −0.69 |
| ‖∇(α·L_unc)‖ / ‖∇L_rec‖ | **0.0001** | 0.39 / 0.40 |
| gradient vào nhánh U | 1.7e-5 | 0.018 / 0.021 |

- **approx (mặc định, đúng eq. 24 + β = 0): U bị tắt ngay từ công thức.** Với β = 0, hàm
  α·L_rec/σ² giảm đơn điệu theo σ², nên tối ưu là σ² → ∞. Adam đi bước chuẩn hoá nên tới
  kẹp chỉ sau ~2 epoch: α·L_unc/L_rec = 0.17 (epoch 1) → 0.027 (epoch 2) → 0.005 (epoch 5)
  → 0.0001. Kẹp ±8 chỉ quyết định σ² dừng ở đâu. Hiện thực đúng nguyên văn eq. (24) thì
  D ≡ C. Table III của paper (w/o U −1.6%) **không thể** đến từ eq. (24) nguyên văn → code
  của tác giả phải giữ σ² hữu hạn (dạng exact eq. 21, β > 0, hoặc σ bị chặn).
- **exact: U làm đúng việc được thiết kế** — σ² phụ thuộc mẫu, giảm trọng số mẫu khó
  (Spearman −0.70), gradient bằng ~40% L_rec — **nhưng không tăng test** (3 seed P+U exact
  0.4524 so với w/o P&U 0.4529). Trên ML-1M (dày), mẫu "khó" là tín hiệu thật chứ không
  phải nhiễu, nên hạ trọng số chúng không có lợi.
- Ngay bảng α của paper (Table V, ML-1M) dao động răng cưa 0.4535–0.4673; α ∈ {0.01, 0.5,
  0.7, 1} còn *thấp hơn* w/o U (0.4599) → hiệu ứng của U trong paper cũng chỉ cỡ nhiễu
  giữa các lần chạy.

### 4.3 Tóm lại: "sai" ở đâu

| | chỗ gây trơ | bằng chứng | cách sửa đã có flag |
|---|---|---|---|
| P | nhiễu chỉ đặt ở x_t của BD-LRU (đọc nguyên văn "replace x_t in (9)–(12)"): residual và nhánh SAM vẫn mang item nguyên vẹn, recurrence pha loãng, cổng học lọc U(0,1) | ΔCE 0.125 → 0.0085 sau khi train với P; r_t 0.50 → 0.17; cos ∇ = cos dropout; không bền hơn với item nhiễu | `--perturbation_site=layer` (P sống lại ở 20 epoch); `--perturbation_gates=clean`, `--perturbation_noise=shuffle` không đủ một mình |
| U | dùng eq. (24) xấp xỉ với β = 0 (mặc định của mình, theo paper) | σ² ở kẹp 99.6%, trọng số 1e-4, gradient 1e-4 | `--uncertainty_loss=exact` (sống, chưa có lợi), `--uncertainty_beta=1` (chưa chạy) |

### 4.4 Việc tiếp theo

```bash
# P: cách hiểu sống sót, chạy đủ (U tắt); --reuse_legacy dùng lại wo_pu / wo_u đã có, không train lại
python scripts/run_variants.py --variants wo_pu wo_u p_layer p_layer_shuffle --reuse_legacy -- --num_workers=2
# U: chỉ U, dạng không suy biến (wo_p chưa có run nào nên sẽ được train)
python scripts/run_variants.py --variants wo_pu wo_p u_exact u_beta u_sequence_beta --reuse_legacy -- --num_workers=2
# nếu p_layer thắng: SLSRec đầy đủ với P ở tầng
python scripts/run_variants.py --variants slsrec slsrec_exact --prefix player -- --perturbation_site=layer --num_workers=2
```
