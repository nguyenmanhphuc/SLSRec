# Vì sao Perturbation/P&U chưa tái lập mức cải thiện của SLSRec?

Ngày phân tích: 08/10/2026. Phạm vi: bài báo và các run ML-1M đã lưu trong repository. Không huấn luyện lại và không thay đổi mô hình.

## Kết luận từ số liệu hiện có

Perturbation có mức cải thiện nhỏ ở run P-only seed 42, và phép đo mới trên checkpoint cho thấy nó đã học tính ổn định trước noise U(0,1). Mức cải thiện ranking thấp hơn Table III. Vấn đề chính có thể là robustness với augmentation này chưa chuyển thành generalization ngoài dữ liệu train. Việc full model không hơn w/o P&U còn liên quan đến uncertainty, lựa chọn checkpoint và biến thiên giữa các run.

Muốn tách hiệu ứng P, phải so **w/o U (C, chỉ P)** với **w/o P&U (B)**. So full SLSRec với B gộp P, U và tương tác giữa hai nhánh.

| Nguồn / cấu hình | Recall@20 | NDCG@20 | Thay đổi tương đối so với B cùng nguồn |
|---|---:|---:|---:|
| Bài báo: w/o P&U | 0.44852 | 0.22980 | — |
| Bài báo: w/o U, chỉ P | 0.45987 | 0.23698 | +2.53% / +3.12% |
| Bài báo: full SLSRec | 0.46732 | 0.24141 | +4.19% / +5.05% |
| Repo seed 42: B_sam_gate | 0.451789 | 0.230957 | — |
| Repo seed 42: C_perturbation | 0.453099 | 0.232126 | +0.29% / +0.51% |
| Repo seed 42: D_paperfig, U approx | 0.451583 | 0.230874 | -0.05% / -0.04% |
| Repo seed 42: D_paper, U approx, eq.20 | 0.448045 | 0.227708 | -0.83% / -1.41% |

Nguồn: `paper/SLSRec.txt:1405` (Table III); `experiments/slsrec/<run>/test_metrics.json`. Các phần trăm trên dùng B làm mẫu số; phần trăm in trong Table III dùng full model làm mẫu số.

Baseline B hiện tại đã gần số bài báo. Khoảng cách chủ yếu nằm ở mức cải thiện của các thành phần bổ sung. Tuy nhiên, B42 và C42 được train trên phiên bản PyTorch/GPU khác nhau, nên đây chưa phải phép so sánh cùng nền tảng.

`C_perturbation_theta0_ml-1m_seed42` vẫn tính hai lượt CE với dropout riêng, nhưng không thay input bằng nhiễu. Nó đạt R20=0.449947, N20=0.230897. C(theta=.1) hơn control này +0.70% R20 và +0.53% N20. C và control dùng cùng môi trường 4080/PyTorch 2.1.0, nhưng quỹ đạo RNG và epoch checkpoint vẫn khác nhau. Kết quả này phù hợp với tác dụng P nhỏ, chưa đủ chứng minh gain ổn định.

Không phải mọi metric đều giảm: D_paperfig đạt R10=0.351066, hơn B42=0.343664 khoảng 2.15%, dù R20/N20 gần như bằng B. Cần xác định đúng metric khi nói không cải thiện.

## 1. Uncertainty approx bị suy biến: bằng chứng mạnh cho full P&U

Đặt z=log(sigma²). Loss mặc định theo eq.24 là:

```text
L = (1 + alpha * exp(-z)) * CE_clean + CE_perturbed
dL/dz = -alpha * exp(-z) * CE_clean < 0
```

Với beta=0, estimator luôn có động lực giảm phần loss của mình bằng cách tăng variance. Đây là hệ quả của objective ghi trong bài, không nhất thiết là lỗi cài đặt. Các tham số estimator dùng chung nên không có nghĩa mọi z tăng đơn điệu từng bước.

Code clamp z trong [-8,8]. Khi giá trị thô vượt +8, gradient qua clamp bằng 0. Trọng số CE bổ sung chỉ còn `0.3*exp(-8)=0.00010064`.

Chẩn đoán đã lưu trên 512 cửa sổ train, ba lượt dropout:

| Phép đo | D_paperfig | D_paper / eq.20 |
|---|---:|---:|
| Vị trí có logvar thô > 8, train mode | 99.564% | 99.244% |
| alpha * L_unc / tổng loss, tại epoch checkpoint | 0.005107% | 0.006613% |

SHA-256 của hai checkpoint hiện tại đã được đối chiếu với JSON chẩn đoán và đều khớp. Nguồn: `audit_2026-10-06/uncertainty_new_runs_vi.md:34`, `uncertainty_fig2_new.json`, `uncertainty_eq20_new.json`.

Vì vậy full model mất phần lớn tác dụng U ở giai đoạn sau. U vẫn ảnh hưởng lúc đầu: đóng góp khoảng 8% tổng loss ở epoch 1. Kết luận này không giải thích trực tiếp mức gain nhỏ của C so với B, vì C đã tắt U.

Chuyển sang `uncertainty_loss=exact` giữ nhánh hoạt động theo softmax temperature, nhưng đây là objective khác approximation eq.24. Kết quả ba seed 42–44 vẫn chưa cho gain ổn định:

| Cấu hình | R20 trung bình ± SD mẫu | N20 trung bình ± SD mẫu |
|---|---:|---:|
| B, w/o P&U | 0.452919 ± 0.002003 | 0.231792 ± 0.001344 |
| D_exact | 0.452371 ± 0.002343 | 0.231517 ± 0.000773 |

D_exact hơn B ở seed42 và thấp hơn ở seed43/44. Ba seed chưa đủ cho kết luận significance.

## 2. Perturbation chỉ tác động một đường thông tin

Trong mỗi SG-LRU layer:

```text
G = SAM(W1*x) + Conv1(W1*x)
H = BD-LRU(perturb(Conv2(W2*x)))
output = LN(Dropout(W3*(G*H)) + x)
```

Nguồn: `model/slsrec.py:168`, `model/bdlru.py:146`. Nhiễu được áp dụng trước cả r_t và i_t, thay toàn bộ vector theo Bernoulli(theta), đúng cách đọc eq.17 và eq.9–12. Padding không bị nhiễu; training có lượt P, evaluation thông thường không có.

Residual x và nhánh tạo G không nhận phép thay nhiễu trực tiếp trong lớp đó. Các lớp sau vẫn có thể chịu ảnh hưởng gián tiếp qua đầu ra lớp trước. Vì vậy mạng có đường giữ thông tin giúp giảm phụ thuộc vào BD-LRU bị nhiễu. Đây là cơ chế hợp lý cần đo mức độ trên checkpoint; cấu trúc đường đi một mình chưa chứng minh mạng thực sự bỏ qua H.

Ở C42 epoch250, CE_clean=4.991699, CE_perturbed=4.999911, chỉ chênh khoảng 0.1645%; mean chênh lệch từ epoch50 là 0.006351. Điều này cho thấy nhiễu gây thêm scalar loss nhỏ. **Loss gần nhau không chứng minh gradient giống nhau**, cũng không chứng minh không có regularization.

Phép đo mới ngày08/10 trên checkpoint C42 và control theta0, không cập nhật tham số và tắt dropout để tách P:

| Noise chèn khi chẩn đoán | Checkpoint train P(theta=.1) | Checkpoint train hai CE(theta=0) |
|---|---:|---:|
| theta=.1: KL(clean || perturbed), mỗi target hợp lệ | 0.00731–0.00756 | 0.0587–0.0646 |
| theta=.1: thay đổi logit L2 tương đối | 3.98–4.11% | 12.1–12.8% |
| theta=.1: overlap top10 tại cuối cửa sổ train | 93.2–94.5% | 82.8–85.0% |
| theta=.1: cosine gradient CE sạch/nhiễu | 0.9861–0.9894 | 0.9057–0.9119 |

Output được đo trên128 cửa sổ train ngẫu nhiên /15,257 nhãn hợp lệ, ba noise seed; gradient trên16 cửa sổ /1563 nhãn hợp lệ, hai noise seed. Overlap top10 được tính sau mask lịch sử trong cửa sổ và item0, không phải Recall@10 của tập test.

Checkpoint C ổn định hơn rõ rệt với cùng theta=.1: **P đã tạo ra robustness nội bộ**. Gradient gần đồng hướng ở checkpoint cuối phù hợp với việc mô hình đã thích nghi với augmentation; không có nghĩa chúng giống nhau suốt training. Với theta=1, checkpoint C có KL khoảng1.085–1.088, logits đổi khoảng47.8%, overlap top10 còn37.7–38.3%, cosine gradient khoảng.260. Do đó không thể kết luận BD-LRU/P hoàn toàn vô tác dụng.

Zero toàn bộ output H của BD-LRU làm CE trên cùng128 cửa sổ tăng từ4.7715 lên6.8450, overlap top10 chỉ còn14.30%. Vì vậy giả thuyết mạnh rằng mô hình bỏ hẳn nhánh H **không được dữ liệu ủng hộ**. Các đường residual/G chỉ là đường thông tin bổ sung, không đủ chứng minh mạng dùng chúng thay thế hoàn toàn recurrence.

## 3. Nhiễu nhân tạo có thể không khớp noise trong dữ liệu

U(0,1) thay thế input đã qua projection và Conv2/SiLU; nó không phải một item tương tác sai lấy từ phân phối dữ liệu. Noise có mean0.5/std0.289 và không có thành phần âm. Probe trên C cho thấy pre-replacement input có mean/std=.382/.754 ở lớp0 và .341/.646 ở lớp1; lớp1 đã chịu ảnh hưởng gián tiếp của P ở lớp0.

So tại cùng các vị trí được chọn để thay thế, với một mẫu uniform mới và giữ tham số gates cố định:

| Lớp | i_t input gốc / uniform | a_t input gốc / uniform | Norm input term b_t gốc / uniform |
|---|---:|---:|---:|
| 0 | .380 / .240 | .878 / .966 | 1.657 / .836 |
| 1 | .290 / .159 | .920 / .978 | .902 / .539 |

Uniform làm input gate nhỏ hơn, hệ số giữ state a_t gần1 hơn, và norm tín hiệu mới nhỏ hơn. Kết quả **phù hợp với cơ chế giảm ảnh hưởng noise và giữ lại state cũ**, nhưng chưa chứng minh mạng học một bộ phân loại noise. Nó có thể giúp giải thích vì sao đạt ổn định với augmentation mà không thu được gain lớn với tương tác thật. Quy kết đây là nguyên nhân của khoảng cách với bài vẫn là giả thuyết. Thử noise khác là nghiên cứu mở rộng, không còn tái lập nguyên văn eq.17.

Bằng chứng và cách chạy lại: `audit_2026-10-08/perturbation_checkpoint_probe.json`, `perturbation_checkpoint_probe.py`. Script nạp checkpoint strict, dùng CPU/PyTorch2.1.0, không optimizer step, ghi riêng JSON chẩn đoán. SHA-256 của checkpoint, dataset và code cùng các chỉ số cửa sổ mẫu nằm trong JSON.

## 4. Điều kiện tái lập và lựa chọn checkpoint còn ảnh hưởng

- B42 và C42 có cùng hparams chính: dropout .2, theta .1, AdamW lr .001 / wd .01, batch128, validation mỗi5 epoch và patience10. Nhưng B42 dùng PyTorch2.0.1/cu117 RTX4050; C42 dùng PyTorch2.1.0/cu118 RTX4080 SUPER.
- P-only mới có một seed. Gain R20 tuyệt đối .00131 nhỏ hơn SD giữa seed của B (.00200); đây là dấu hiệu cần nhiều seed, không phải phép kiểm significance.
- Chọn checkpoint bằng R10, nhưng Table III báo R20/N20. B42 chọn epoch225, C42 chọn250, theta0 chọn120; thứ tự thắng thua có thể đổi theo epoch.
- Các run `gateexact_*` cũ validate theo iteration, khác các run B/C mới. Không xem chúng như ablation chỉ thay P&U.
- Eq15/16 mô tả last-position loss; harness dùng CE mọi vị trí. Tác giả nói theo LRURec nhưng không làm rõ điểm này. Đây là giả định ảnh hưởng cách học, xem `reproduction_report.md:294`.
- Các lựa chọn kernel, initialization, xử lý padding, weight decay của lambda không được bài mô tả đủ để xác nhận khớp hoàn toàn. Chúng ảnh hưởng kiến trúc chung; chưa có bằng chứng rằng một lựa chọn cụ thể gây mất gain của P.
- Mask toàn bộ lịch sử thay vì cửa sổ200 làm R20 của một D checkpoint tăng khoảng1.87% trong probe cũ. Nó có thể góp phần vào chênh lệch tuyệt đối với bài, nhưng cần đánh giá B và C cùng giao thức để kết luận về delta của P.

Không nên dùng status cũ trong README/report để kết luận chỉ có smoke run: hiện repo đã có các run hội tụ. `D_paper`, `D_papereq`, `D_papereqfix` có cùng trajectories/metrics, không phải ba lần chạy độc lập.

## 5. Bài báo cũng không cho thấy P-only luôn có lợi

Theo Table III, trên Sports, w/o U (P-only) có R20=.04309, thấp hơn w/o P&U=.04436 khoảng2.86%; N20 giảm4.36%. Full model lại tốt hơn B. Như vậy tương tác P–U có thể quan trọng, và không thể đọc kết quả full như bằng chứng mọi tác dụng đến từ P.

Bài không báo seed, SD hay significance của ablation. Đồ thị chứng minh overfitting tập trung ở Beauty/Sports; chưa đủ để suy ra ML-1M trong protocol hiện tại cần cùng mức regularization.

## Thí nghiệm ưu tiên tiếp theo

1. Cùng môi trường, initialization/backbone và lịch training: chạy B, hai CE theta0, C(theta=.1) qua ít nhất5 seed. Lưu generator riêng cho shuffle/dropout/perturbation nếu muốn kiểm soát trajectory, và dùng validation để chọn checkpoint.
2. Mở rộng probe output/gradient và routing đã thực hiện sang B, nhiều seed, các checkpoint đầu/giữa training và dữ liệu held-out; tách dropout để không nhầm khác biệt dropout với P.
3. Hoàn thiện thiết kế 2×2: không P/U, P-only, U-only, P+U; giữ `approx` và `exact` thành hai nhóm objective riêng.
4. Nếu P-only vẫn yếu, thử theta và dropout theo grid với ngân sách tìm kiếm tương đương giữa các ablation. Không tự tăng theta vì perturbation quá mạnh có thể làm mất tín hiệu.

Các assertion “falsified” trong `tests/test_perturbation_is_inert.py` chưa đủ bác bỏ bài báo: học thuộc8 chuỗi không phủ định generalization; loss bằng nhau không phủ định gradient khác; ngưỡng KL/cosine tự chọn không phải điều kiện cần cho regularization hiệu quả.
