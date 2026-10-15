# Báo cáo tiến độ ý tưởng KAPTRec

*Context-Guided Adaptive Perturbation with KAN-Gated Transition Fusion for Sequential Recommendation*

Cập nhật: 15/10/2026. Nguồn ý tưởng: slide `Context-Guided Adaptive Perturbation with KAN-Gated Transition Fusion for Sequential Recommendation.pptx` (01/09/2026). Backbone: bản tái lập SLSRec trong repo này.

---

## 1. Tóm tắt

| Đóng góp | Trạng thái | Kết luận hiện tại |
|---|---|---|
| **C1** Full-KAN gate | Đã chạy (ML-1M 1 seed, Beauty 2 seed cho biến thể `kan`) | **Không đạt.** Mọi phép fusion bỏ tích G ⊙ H đều thua backbone trên ML-1M. KAN còn thua MLP cùng cấu trúc. |
| **C2** BD-LRU nhận biết chuyển tiếp | Đã hiện thực, chưa có kết quả | Chưa biết |
| **C3** Perturbation thích nghi + JS | ML-1M 1 seed, Beauty 1 seed | **Triển vọng nhất.** Beauty +4.8% R@20 / +6.0% N@20 so với backbone, gần chạm SLSRec đầy đủ trong bài. ML-1M ≈ 0. |
| Uncertainty theo toàn lịch sử | Đã hiện thực, chưa có kết quả | Chưa biết |

**Câu chuyện đang hình thành:** trong SG-LRU, **tích G ⊙ H là thành phần thiết yếu**, còn nội dung của G (SAM, Conv, SiLU…) hầu như không quan trọng. Đóng góp có giá trị nằm ở **điều chuẩn hóa (C3)**, và nó chỉ phát huy ở dữ liệu overfit (Beauty), không phát huy ở dữ liệu không overfit (ML-1M).

---

## 2. Bối cảnh: các phát hiện từ tái lập SLSRec dẫn tới ý tưởng

| Thành phần SLSRec | Bài báo công bố (Beauty) | Đo được trong repo | Nguồn |
|---|---|---|---|
| Backbone w/o P&U | 0.07345 / 0.03589 | **0.07380 / 0.03583** (3 seed, dropout 0.6) | `noP_d06_beauty_seed{42,2026,100}` |
| Đóng góp của SAM | −6.2% / −5.0% nếu bỏ | ≈ 0 (backbone 3 seed, full 2 seed) | `*_noSAM_*` |
| Đóng góp của P (U(0,1)) | +3.25% / +2.59% | +0.8% / +0.8% (3 seed) | `P_only_d06_*`, `C0_d06_*` |
| Đóng góp của U (Gibbs) | +1.9% / +2.9% | ≈ 0; eq. 24 với β = 0 suy biến | `full_exact_d06_*` |
| Dropout tối ưu | 0.75 | ≈ 0.6 cho mọi biến thể | |

Hai điểm yếu của P gốc đã đo được và trở thành động lực cho C3:
1. **Nhiễu U(0,1) dễ bị nhận ra.** Gate của BD-LRU học cách lọc bỏ nó. Với P gốc, L_per − L_rec chỉ khoảng +0.0025, tức nhiễu gần như vô hại.
2. **Lượt forward thứ hai không có ràng buộc giữa hai view.** Phần lợi ích ở dropout 0.75 chủ yếu đến từ việc có thêm một mẫu dropout.

Beauty overfit rất nặng (R@20 trên item đã train / item chưa thấy = **6.0×**, so với **1.14×** trên ML-1M), nên đây là nơi kiểm regularizer có ý nghĩa nhất.

---

## 3. Ý tưởng KAPTRec (theo slide)

**C1, Full-KAN gate (chọn nhánh phi tuyến)**
> Qₜ = [LN(Gₜ) ‖ LN(Hₜ)], oₜ = σ(KAN_{2d→d}(Qₜ)), Fₜ = oₜ ⊙ Hₜ + (1 − oₜ) ⊙ Gₜ, Zₜ = LN(Xₜ + W_z Fₜ).
> Slide xếp MLP gate làm ablation hiệu năng.

**C2, BD-LRU nhận biết chuyển tiếp**
> qₜ = [xₜ₋₁, xₜ, xₜ − xₜ₋₁, xₜ₋₁ ⊙ xₜ], rₜ = σ(MLP(qₜ)); phần còn lại giữ phép đệ quy ổn định của BD-LRU.

**C3, perturbation thích nghi trên embedding đầu vào**
> πₜ = π_max σ(MLP(stopgrad fₜ)), mₜ ~ Bernoulli(πₜ) (Gumbel-sigmoid có straight-through);
> εₜ = ε_min + (ε_max − ε_min) σ(MLP_ε(stopgrad fₜ));
> d_ctx = (stopgrad aₜ − xₜ)/‖·‖ với aₜ = W_c Zₜ; d_rnd = sign(xₜ) ⊙ ξ/‖ξ‖; dₜ = normalize((1 − ρ) d_ctx + ρ d_rnd);
> x̃ₜ = xₜ + mₜ εₜ dₜ;
> L = L_rec^clean + λ_p L_rec^pert + λ_c ω_u JS(p^clean ‖ p^pert) + α L_unc.

**Uncertainty theo toàn lịch sử**
> αₜ = softmax(vᵀ tanh(W_z Zₜ + W_y e_y)), c_u = Σ αₜ Zₜ, d̄_u = Σ αₜ[1 − cos(Gₜ, Hₜ)], r̄_u = Σ αₜ[1 − cos(Zₜ₋₁, Zₜ)],
> σ_u = softplus(MLP[c_u ‖ Z_L ‖ e_y ‖ d̄_u ‖ r̄_u]).

---

## 4. Hiện thực

Mọi thành phần đều là cờ mới, **mặc định tắt**, nên kết quả của SLSRec gốc không thay đổi.

| Thành phần | Cờ | File |
|---|---|---|
| C1 | `--fusion_mode=kan_gate` | `model/kan.py` |
| C1, ablation MLP | `--fusion_mode=mlp_gate` | `model/kan.py` |
| KAN thay hẳn G ⊙ H (thử nghiệm phụ) | `--fusion_mode=kan` | `model/kan.py` |
| G ⊙ H cộng KAN hiệu chỉnh (thử nghiệm phụ) | `--fusion_mode=kan_res` | `model/kan.py` |
| C2 | `--bdlru_transition=True` | `model/bdlru.py` |
| C3 | `--kap_perturb=True` (và các cờ `--kap_*`) | `model/kaptrec.py`, `trainer/slsrec.py` |
| Uncertainty theo lịch sử | `--kap_uncertainty=True` | `model/kaptrec.py` |

Tham số mặc định của C3: π_max = 0.3, π_target = 0.1, budget weight = 10, ε ∈ [0.05, 0.5] (tương đối so với ‖xₜ‖), ρ = 0.5, τ = 0.5, λ_p = λ_c = 1, α_unc = 0.1.

### 4.1 Các điểm slide chưa nói rõ và cách đã chọn

| Điểm mơ hồ | Lựa chọn | Lý do |
|---|---|---|
| Cái gì train bộ điều khiển C3 (π, ε, W_c) | **Đối kháng qua gradient reversal** (`--kap_controller=adv`), cộng số hạng ngân sách (mean π − π_target)². Cách đọc ngây thơ giữ lại ở `--kap_controller=min`. | Nếu bộ điều khiển cùng giảm loss với encoder, nó sẽ học cách tạo nhiễu vô hại (π → 0, ε → ε_min). Đây cũng là kiểu sụp đổ đã làm U của SLSRec mất tác dụng. |
| Thang đo của ε | Tương đối: δ = m · ε · ‖xₜ‖ · d | Không phụ thuộc thang đo của embedding |
| Đặc trưng fₜ | [e_y(t), Zₜ, 1 − cos(Gₜ, Hₜ), 1 − cos(Zₜ₋₁, Zₜ), t/L], chặn gradient | Hiện thực hóa "target, branch, transition, t" của slide |
| Dạng của L_unc | CE_u/(2σ²) + log σ, ở vị trí cuối | Có điểm tối ưu hữu hạn, khác eq. 24 của SLSRec |
| ω_u | stopgrad(σ⁻²), chuẩn hóa về trung bình 1 | Để σ không thể "né" số hạng JS |
| Single-layer KAN không biểu diễn được tích | `kan` dùng 2 lớp KAN; `kan_gate` 1 lớp như slide | Một lớp KAN chỉ là tổng các hàm một biến |
| Dòng đầu vào toàn padding (user 1 item) | Trọng số pooling = 0 | Tránh NaN, vì harness có các dòng như vậy |

### 4.2 Kiểm thử

- `tests/test_kan_fusion.py`: 9 test.
- `tests/test_kaptrec.py`: 20 test. Kiểm tính nhân quả, bất biến với padding, chỉ áp dụng lúc train, giới hạn của π và ε, padding không bị nhiễu, mặt nạ 0/1 bám theo π, gradient reversal, và các dòng không có lịch sử.
- Bộ test tổng: chỉ còn 3 test fail có sẵn từ trước, liên quan tới cờ uncertainty chưa nối dây (`test_alt_readings.py`), không liên quan tới KAPTRec.

### 4.3 Chi phí (một batch Beauty, CPU, so với backbone)

| Biến thể | Thời gian | Tham số encoder (không tính embedding) |
|---|---|---|
| Backbone G ⊙ H | 1.0× | 316k |
| C1 `kan_gate` | khoảng 1.3× | 907k |
| C1 `mlp_gate` | — | 416k |
| `kan` | khoảng 2.0× | 1.20M |
| C2 | khoảng 0.8× | 480k |
| **C3** | **khoảng 3.7×** (GPU thực tế: 5.5 giờ so với khoảng 1.5–2 giờ trên Beauty) | 316k khi suy luận; bộ điều khiển thêm 21k tham số, chỉ dùng lúc train |

---

## 5. Giao thức thực nghiệm

- **Backbone so sánh:** SLSRec với P và U gốc tắt. Beauty dùng dropout 0.6 (mức tái lập khớp bài), ML-1M dùng dropout 0.2.
- **Lịch train:** theo bài (1000 epoch, validate mỗi 5 epoch, patience 10, chọn checkpoint theo val R@10).
- **Ghép cặp theo seed:** mỗi biến thể so với backbone cùng seed.
- **Ngưỡng có ý nghĩa với một seed** (đo từ dao động giữa các seed):
  - ML-1M: khoảng **±0.5%** mỗi run, nên chênh **trên khoảng 1%** mới đáng tin.
  - Beauty: khoảng **±1–2%** mỗi run, nên cần **trên khoảng 2%** trên validation hoặc **trên khoảng 3%** trên test.
- **Thước đo phụ:** trung bình val R@10 / N@10 trên vùng đi ngang (epoch ≥ 50), ít nhiễu hơn checkpoint tốt nhất.
- **Công cụ:** `scripts/compare_runs.py` (so sánh), `scripts/eval_checkpoint.py` (chấm lại test từ checkpoint), `scripts/fit_check.py` (đo mức học thuộc).

---

## 6. Kết quả

### 6.1 C1 và các phép fusion trên ML-1M (seed 42, P và U tắt)

| Phép fusion | R@20 | N@20 | Val R@10 | Val N@10 | Best / dừng ở epoch |
|---|---|---|---|---|---|
| **G ⊙ H** (backbone) | 0.45179 | 0.23096 | 0.3669 | 0.2159 | 225 / 275 |
| `mlp_gate` (trộn lồi, MLP) | −0.8% | −1.5% | **−1.5%** | **−1.3%** | 95 / 145 |
| **`kan_gate` (C1 đúng như slide)** | −1.5% | −1.2% | **−2.8%** | **−3.4%** | 90 / 140 |
| `kan` (KAN thay hẳn tích) | −4.5% | −3.3% | **−3.6%** | **−4.8%** | 65 / 115 |
| `kan_res` (tích + KAN hiệu chỉnh) | chưa có | | | | |
| *LRURec (để đối chiếu)* | −3.0% | −3.0% | −3.5% | −4.6% | |

### 6.2 KAN thay tích trên Beauty (dropout 0.6, 2 seed)

| `kan` so với backbone | R@1 | R@20 | N@20 | R@50 | Val R@10 | Val N@10 |
|---|---|---|---|---|---|---|
| seed 42 | +10.0% | −3.3% | +0.1% | −5.1% | −0.2% | +0.9% |
| seed 2026 | +9.2% | −1.2% | +1.4% | −3.1% | +0.7% | +3.6% |

Trên Beauty, KAN dịch kết quả về phía đầu danh sách (R@1 tăng khoảng 9–10%, R@50 giảm 3–5%), còn R@20 / N@20 không cải thiện rõ.

### 6.3 C3 trên ML-1M (seed 42, U tắt)

| | R@20 | N@20 | Val R@10 | Val N@10 |
|---|---|---|---|---|
| Backbone | 0.45179 | 0.23096 | 0.3669 | 0.2159 |
| C0 (2 lượt forward, θ = 0) | −0.4% | 0.0% | −1.2% | −1.6% |
| P gốc của SLSRec | +0.3% | +0.5% | +0.1% | −0.7% |
| **C3** | **+0.3%** | **+0.4%** | **+0.5%** | **−0.1%** |

Thống kê lúc train của C3: π trung bình ổn định ở **khoảng 0.118** (ngân sách 0.1); L_pert − L_rec **khoảng +0.05**, gấp khoảng 20 lần P gốc; JS khoảng 0.06. **Bộ điều khiển hoạt động đúng thiết kế**, nhưng ML-1M không overfit nên không có gì để cải thiện.

### 6.4 C3 trên Beauty (seed 42, dropout 0.6, U tắt)

| | R@1 | R@10 | N@10 | **R@20** | **N@20** | R@50 | Val R@10 | Val N@10 |
|---|---|---|---|---|---|---|---|---|
| Backbone | 0.01260 | 0.05331 | 0.03044 | 0.07336 | 0.03548 | 0.10958 | 0.0655 | 0.0379 |
| C0 | 0.01334 | 0.05303 | 0.03061 | 0.07383 | 0.03585 | 0.10914 | −1.4% | −2.2% |
| P gốc | 0.01295 | 0.05364 | 0.03086 | 0.07493 | 0.03623 | 0.10965 | +0.7% | −0.3% |
| **C3** (checkpoint epoch 145) | **0.01411** | **0.05576** | **0.03229** | **0.07689** | **0.03761** | **0.11226** | **+5.8%** | **+5.6%** |

| C3 so với | R@20 | N@20 |
|---|---|---|
| Backbone | **+4.8%** | **+6.0%** |
| P gốc của SLSRec | +2.6% | +3.8% |
| LRURec (3 seed) | +7.5% | +6.2% |
| SLSRec đầy đủ trong bài (0.07727 / 0.03790) | −0.5% | −0.8% |

Thống kê lúc train: π khoảng 0.15–0.16, L_pert − L_rec khoảng +0.30, JS khoảng 0.28. Nhiễu trên Beauty "mạnh" hơn trên ML-1M khoảng 5 lần.

**Lưu ý về số liệu:** `test_metrics.json` của run này bị hỏng (R@20 = 0.00036), vì hai tiến trình cùng chạy vào một thư mục và tiến trình sau ghi đè `best_acc_model.pth` bằng model chưa train. Số ở bảng trên được chấm lại bằng `scripts/eval_checkpoint.py` trên checkpoint **cuối** (epoch 145) của tiến trình đã train trọn. Script này đã được kiểm chứng: nó tái tạo đúng từng chữ số kết quả test của `kap_C3_ml-1m_seed42`. Checkpoint tốt nhất theo validation (epoch 95) không còn, nên con số này **chưa theo đúng protocol chọn checkpoint** và cần chạy lại sạch.

### 6.5 Chưa có kết quả

C2 (`kap_C2_*`), C1 trên Beauty (`kap_C1_*`, `kap_C1mlp_*`), uncertainty (`kap_C3U_*`), bản đầy đủ (`kap_full_*`), `noP_kanres_ml-1m_seed42`, các seed lặp lại và ablation của C3.

---

## 7. Phân tích

1. **Tích G ⊙ H là cốt lõi của SG-LRU.** Trên ML-1M, càng xa phép nhân thì càng thua: trộn lồi bằng MLP −1.5%, bằng KAN −2.8% đến −3.4%, KAN thay hẳn −3.6% đến −4.8% (validation). Kết hợp với kết quả trên Beauty (bỏ SAM, chỉ SAM, gate SiLU, gate theo token, sparsemax, entmax: đều trong ±2%): **nội dung của G không quan trọng, nhưng việc G nhân vào H thì có.** Điều này đi ngược giả định của C1 rằng phép fusion cố định là điểm yếu.
2. **KAN không có lợi thế so với MLP** ở vai trò gate, dù tốn khoảng 6 lần tham số cho phần gate.
3. **Regularizer chỉ giúp nơi có overfit.** C3 xấp xỉ 0 trên ML-1M (seen/unseen 1.14×), nhưng +5–6% trên Beauty (6.0×). P gốc thì không giúp ở cả hai, vì nhiễu của nó gần như vô hại.
4. **Lợi ích của C3 không đến từ lượt forward phụ.** C0 (cùng số lượt forward, không nhiễu) không cải thiện. Phần lợi ích đến từ nhiễu đối kháng thích nghi cộng ràng buộc JS. Thành phần nào trong đó quan trọng thì phải chờ ablation.

---

## 8. Sự cố dữ liệu và bài học quy trình

- **Chạy trùng tên thư mục** đã xảy ra nhiều lần (`noP_d06_*`, `noP_d06_bias_*`, `full_exact_d06_*`, `P_only_d06_seed100`, `kap_C3_beauty_seed42`). Trainer ghi log theo kiểu nối thêm và ghi đè checkpoint, nên một tiến trình chạy sau có thể làm hỏng kết quả của tiến trình trước. **Nên thêm chốt chặn trong `train.py`**: từ chối khởi động nếu thư mục `--exp_name` đã có kết quả.
- **Copy lồng thư mục** (`experiments/slsrec/slsrec/…`): `compare_runs.py` đã được sửa để tìm cả trong thư mục con.
- **Cờ bị bỏ qua âm thầm:** các cờ `--perturbation_site/_gates/_noise` từng có trong `config.py` nhưng chưa nối vào model. Đã sửa. Hai cờ `--uncertainty_granularity` và `--uncertainty_weight_norm` vẫn chưa được nối dây: **đừng dùng**.

---

## 9. Đánh giá từng đóng góp

| Đóng góp | Đánh giá | Khuyến nghị |
|---|---|---|
| C1 (KAN gate) | Thua backbone, KAN thua MLP | **Không dùng làm đóng góp chính.** Nếu `kan_res` cũng không giúp, đưa kết quả C1 vào bài như một ablation phủ định: "fusion phi tuyến không thay được tích G ⊙ H". |
| C2 | Chưa có dữ liệu | Chạy trên Beauty và ML-1M |
| **C3** | +4.8% / +6.0% trên Beauty (1 seed); ≈ 0 trên ML-1M | **Ứng viên đóng góp chính.** Cần 3 seed, ablation thành phần, và thêm một dataset thưa (Sports). |
| Uncertainty theo lịch sử | Chưa có dữ liệu | Chạy sau khi C3 được xác nhận |

Để C3 đứng vững trước reviewer, cần thêm:
1. ≥ 3 seed trên Beauty, với chênh lệch ghép cặp nhất quán.
2. Ablation: không JS, bộ điều khiển `min`, chỉ hướng ngẫu nhiên.
3. Một baseline điều chuẩn đã biết, ví dụ R-Drop hoặc nhiễu embedding ngẫu nhiên cộng JS, để chứng minh tính thích nghi có giá trị.
4. Một dataset thưa khác (Sports) để chứng minh tính tổng quát.

---

## 10. Việc tiếp theo và lệnh

Mọi lệnh chạy trên container, mỗi run một tên riêng.

**1. Chạy lại sạch C3 seed 42**, nếu tiến trình trùng đã dừng:

```bash
python train.py --dataset_code=beauty --model_code=slsrec --use_perturbation=False --use_uncertainty=False --slsrec_dropout=0.6 --kap_perturb=True --seed=42 --exp_name=kap_C3_beauty_seed42_r2
```

**2. Seed lặp lại:**

```bash
for s in 2026 100; do python train.py --dataset_code=beauty --model_code=slsrec --use_perturbation=False --use_uncertainty=False --slsrec_dropout=0.6 --kap_perturb=True --seed=$s --exp_name=kap_C3_beauty_seed$s; done
```

**3. Ablation của C3** (mỗi lệnh một terminal):

```bash
python train.py --dataset_code=beauty --model_code=slsrec --use_perturbation=False --use_uncertainty=False --slsrec_dropout=0.6 --kap_perturb=True --kap_lambda_c=0 --seed=42 --exp_name=kap_C3_noJS_beauty_seed42
python train.py --dataset_code=beauty --model_code=slsrec --use_perturbation=False --use_uncertainty=False --slsrec_dropout=0.6 --kap_perturb=True --kap_controller=min --seed=42 --exp_name=kap_C3_min_beauty_seed42
python train.py --dataset_code=beauty --model_code=slsrec --use_perturbation=False --use_uncertainty=False --slsrec_dropout=0.6 --kap_perturb=True --kap_rho=1.0 --seed=42 --exp_name=kap_C3_rndonly_beauty_seed42
```

**4. C2 và `kan_res`:**

```bash
python train.py --dataset_code=beauty --model_code=slsrec --use_perturbation=False --use_uncertainty=False --slsrec_dropout=0.6 --bdlru_transition=True --seed=42 --exp_name=kap_C2_beauty_seed42
python train.py --dataset_code=ml-1m --model_code=slsrec --use_perturbation=False --use_uncertainty=False --fusion_mode=kan_res --seed=42 --exp_name=noP_kanres_ml-1m_seed42
```

**So sánh:**

```bash
python scripts/compare_runs.py noP_d06_beauty_seed42 P_only_d06_beauty_seed42 kap_C3_beauty_seed42_r2 kap_C3_noJS_beauty_seed42 kap_C3_min_beauty_seed42 kap_C3_rndonly_beauty_seed42 kap_C2_beauty_seed42 --base noP_d06_beauty_seed42 --ref 0.07727 0.03790
```

**Chấm lại test từ checkpoint** khi `test_metrics.json` bị hỏng:

```bash
python scripts/eval_checkpoint.py experiments/slsrec/<run> --ckpt checkpoint-recent.pth
```
