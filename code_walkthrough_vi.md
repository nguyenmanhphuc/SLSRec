# Giải thích toàn bộ code — SLSRec reproduction (`E:\ML\SLSRec`)

> Tài liệu này đi kèm `README.md` và `reproduction_report.md` đã có sẵn trong repo.
> `reproduction_report.md` trả lời câu hỏi "mapping công thức nào ↔ code nào, kết quả đo
> được ra sao". Tài liệu này trả lời câu hỏi khác: **code thực sự chạy như thế nào, dòng
> nào làm gì, và vì sao thiết kế lại như vậy** — kèm phần toán và một góc nhìn phản biện
> về bản thân phương pháp SLSRec.

---

## 0. Bối cảnh: repo này là gì

Đây **không phải** một model tự nghĩ ra, mà là bản **tái hiện (reproduction)** trung
thực của bài báo:

> Yang, Bi, Huang, Li — *"Self-Attention Gated Linear Recurrent Units for Sequential
> Recommendation" (SLSRec)*, IEEE TCSS, DOI `10.1109/TCSS.2026.3679563`.

Ba nguồn được trộn lại có chủ đích:

- **LRURec** (`https://github.com/yueqirex/LRURec`) — cho toàn bộ *harness*: đọc data
  ML-1M, dataloader, vòng lặp train/val/test, checkpoint, tính Recall/NDCG/MRR. Phần
  này **giữ nguyên không đổi** (`model/lru.py`, `trainer/lru.py`, `dataloader/*`,
  `datasets/*` — reproduction_report.md xác nhận bằng `diff` với repo gốc).
- **RecBLR** (`https://github.com/chengkai-liu/RecBLR`) — chỉ dùng làm **tài liệu tham
  chiếu cách implement** cho phần "Behavior-Dependent LRU" (BD-LRU): cách tham số hoá
  recurrence sao cho ổn định, causal depth-wise conv, parallel scan.
  `model/triton_scan.py` là copy nguyên văn kernel Triton của RecBLR.
  `model/bdlru.py` là bản port sang giao diện tensor của LRURec.
- **model/slsrec.py + trainer/slsrec.py** — phần *mới*, hiện thực chính xác các công
  thức (1)–(24) trong paper SLSRec.

Vì sao trộn ba nguồn thay vì viết từ đầu? Vì bài báo tự nói "chúng tôi theo sát cấu hình
của LRURec và xin bộ dữ liệu trực tiếp từ tác giả LRURec" — nên harness (data
split, optimizer, đánh giá) phải đúng y hệt LRURec để so sánh công bằng; còn cơ chế
recurrence có tính ổn định số học tinh vi (eq. 9–12) thì paste lại logic đã được
RecBLR kiểm chứng là an toàn hơn tự suy ra từ công thức paper (vốn có mấy chỗ mơ hồ,
xem mục 9).

---

## 1. Luồng chạy tổng thể — `train.py` là entrypoint

```python
def train(args, export_root=None):
    seed_everything(args.seed)
    ...
    train, val, test = dataloader_factory(args)   # -> dataloader/__init__.py
    model = model_factory(args)                   # -> model/__init__.py
    trainer = trainer_factory(args, model, train, val, test, export_root, args.use_wandb)
    trainer.train()
    return trainer.test()

if __name__ == "__main__":
    set_template(args)   # -> config.py, điền toàn bộ default còn thiếu
    train(args)
```

Đây là pattern **factory + registry** rất phổ biến trong các repo recsys học thuật
(LRURec, SASRec, BERT4Rec đều theo pattern này):

```python
MODELS   = {'lru': LRU, 'slsrec': SLSRec}                 # model/__init__.py
TRAINERS = {'lru': LRUTrainer, 'slsrec': SLSRecTrainer}   # trainer/__init__.py
DATASETS = {'ml-1m': ML1MDataset, 'beauty': BeautyDataset, ...}  # datasets/__init__.py
```

`--model_code` chọn model, `--dataset_code` chọn dataset — tất cả các *ablation*
(A/B/C/D) đều dùng **chung một class** `SLSRec`, khác nhau chỉ ở cờ config
(`--use_sam_gate`, `--use_perturbation`, `--use_uncertainty`, ...). Đây là điểm quan
trọng để hiểu toàn bộ repo: **không có 4 model riêng biệt**, chỉ có 1 kiến trúc với các
nhánh có thể bật/tắt — nhờ vậy `tests/test_tiny_overfit.py` mới chứng minh được "cả 4
ablation đều overfit được 8 chuỗi tổng hợp" bằng cách gọi cùng một constructor với
`overrides` khác nhau.

---

## 2. Tầng cấu hình — `config.py`

File này định nghĩa **một `argparse.ArgumentParser` toàn cục** (`args` là biến module,
được parse ngay khi import — hơi khác thường nhưng là style gốc của LRURec). Ba hàm
đáng chú ý:

- `set_template(args)`: điền các default phụ thuộc dataset (max_len, val_iterations,
  batch size...) **chỉ khi user chưa truyền cờ đó qua CLI** — đây là một patch so với
  LRURec gốc (bản gốc ghi đè vô điều kiện, khiến `--bert_max_len` truyền tay bị "nuốt
  mất", ghi rõ trong comment).
- `set_slsrec_template(args, user_bert_dropout)`: set riêng cho SLSRec — dropout theo
  dataset (0.2 cho ML-1M dày đặc, 0.75 cho Beauty/Sports thưa), `uncertainty_alpha`
  (0.3 mặc định, 0.01 riêng cho Beauty).
- `args, _unknown_args = parser.parse_known_args()` thay vì `parse_args()` — lý do nêu
  rõ trong comment: để `import config` không crash khi chạy dưới `pytest` (vốn có
  `sys.argv` riêng của nó).

Có khoảng 25 cờ mới cho SLSRec, mỗi cờ ứng với đúng một quyết định kiến trúc/toán học
(ví dụ `--bdlru_c` chính là hằng số `c=8` trong eq. 11, `--perturbation_probability`
chính là `θ` trong eq. 17). Đây là thiết kế tốt: **mọi giả định mơ hồ của paper đều lộ
ra thành một cờ CLI có thể bật/tắt để thí nghiệm**, thay vì hard-code một lựa chọn.

---

## 3. Tầng dữ liệu — `datasets/` + `dataloader/`

### 3.1 `datasets/base.py` + `datasets/ml_1m.py`

Pipeline chuẩn kiểu SASRec/BERT4Rec/LRURec:

1. Tải `ratings.dat` (`uid, sid, rating, timestamp`).
2. `remove_immediate_repeats` — bỏ tương tác lặp lại liên tiếp cùng item.
3. `filter_triplets` — lọc 5-core: chỉ giữ user/item có ≥ `min_uc`/`min_sc` = 5 tương
   tác.
4. `densify_index` — remap id gốc → id liên tục `1..N` (id `0` dành riêng cho padding).
5. `split_df` với `split='leave_one_out'`: sort theo `(timestamp, sid)`, item **cuối
   cùng** → test, item **áp chót** → val, phần còn lại → train.
6. Cache kết quả vào `data/preprocessed/<hash cấu hình>/dataset.pkl` (pickle) để lần
   sau không phải xử lý lại — đây là lý do log thấy "Already preprocessed. Skip
   preprocessing".

### 3.2 `dataloader/sas.py` — 3 loại `Dataset` khác nhiệm vụ

- **`SASTrainDataset`**: với mỗi user, cắt cửa sổ trượt (`sliding_window_size`, mặc
  định = 1 tức không trượt) độ dài `max_len`, **left-pad bằng 0**. `tokens` là chuỗi
  input, `labels` là chuỗi bị lệch 1 vị trí (next-item ở *mọi* vị trí, không chỉ vị trí
  cuối) — đây chính là điểm mấu chốt trong mục 8.2 của `reproduction_report.md`: paper
  eq. (15)/(16) chỉ định nghĩa loss ở vị trí cuối, nhưng harness LRURec train theo
  **per-position next-item** (giống causal LM), và code này giữ nguyên convention đó.
- **`SASValidDataset`** / **`SASTestDataset`**: đưa toàn bộ lịch sử (test còn nối thêm
  item validation vào cuối input) và **chỉ đánh giá ở vị trí cuối cùng** — đúng với
  eq. (15) của paper (dự đoán item tiếp theo dựa trên toàn bộ chuỗi quan sát được).
- Đánh giá dùng **full-sort**: chấm điểm toàn bộ `|I|+1` item (không sample negative),
  sau đó gán `-1e9` cho các item **đã xuất hiện trong input** và cho id `0` (padding)
  trước khi rank — xem `calculate_metrics` ở mục 5.

**Vì sao padding bên trái (left padding) chứ không phải bên phải?** Vì đó là quy ước
BERT4Rec/LRURec kế thừa: giữ item cuối cùng "cố định" ở vị trí cuối tensor bất kể chuỗi
dài ngắn ra sao, để suy luận at test-time đơn giản (`x[:, -1]` luôn là dự đoán kế tiếp
đúng cho vị trí "hiện tại"). Hệ quả kéo theo: embedding vị trí, self-attention mask, và
conv nhân quả trong SLSRec đều phải được thiết kế "bất biến với padding trái" — đây là
lý do bạn thấy code liên tục nhân với `mask` ở gần như mọi module (xem mục 4).

---

## 4. Tầng mô hình — `model/`

### 4.1 `model/lru.py` — LRU gốc (baseline, không đổi)

Đây là bản LRU-based sequential recommender gốc của LRURec, dùng **recurrence số phức**
(complex-valued linear recurrence), khác hẳn BD-LRU thực số ở mục 4.2. Ý tưởng nền tảng
đến từ bài **"Resurrecting Recurrent Neural Networks for Long Sequences"** (Orvieto et
al., ICML 2023) — đề xuất dùng recurrence tuyến tính với eigenvalue phức nằm trong
vòng tròn đơn vị, tham số hoá qua `(ν, θ, γ)`:

```
λ = exp(-exp(ν) + i·exp(θ))      # số phức, |λ| < 1 -> ổn định
h_t = λ ⊙ h_{t-1} + γ ⊙ (B x_t)  # γ chuẩn hoá để Var(h) ổn định
```

`LRULayer.lru_parallel` implement **thuật toán scan song song kiểu "divide-and-conquer"
gấp đôi mỗi bước** (chèn link tới bài blog gốc ngay trong code:
`kexue.fm/archives/9554`), yêu cầu độ dài chuỗi là luỹ thừa của 2 — đó là lý do
`LRUModel.forward` phải pad `seq_len` lên `2**⌈log2(L)⌉` trước khi đưa vào block.

Bạn không cần sửa file này — nó chỉ tồn tại để **baseline `python train.py
--dataset_code=ml-1m` (không kèm `--model_code=slsrec`) chạy được y hệt repo LRURec
gốc**, dùng làm đối chứng.

### 4.2 `model/bdlru.py` — trái tim của SLSRec: BD-LRU

Đây là phần quan trọng nhất để hiểu SLSRec. Khác với LRU số phức ở trên, BD-LRU dùng
**recurrence thực, gọi là "input-dependent" hay "behavior-dependent"**: hệ số suy giảm
`a_t` không cố định mà phụ thuộc vào chính input tại từng bước thời gian.

**Bước 1 — hai cổng (gates), eq. (9)/(10):**

```python
recurrence, inp = self.gates(x).chunk(2, dim=-1)   # 1 Linear duy nhất, tách làm 2
r_t, i_t = torch.sigmoid(recurrence), torch.sigmoid(inp)
```

`r_t` là "recurrence gate" (quyết định giữ lại bao nhiêu quá khứ), `i_t` là "input
gate" (quyết định cho bao nhiêu input mới vào) — về mặt trực giác **tương tự cổng quên
(forget gate) và cổng vào (input gate) của LSTM**, nhưng ở đây không có nonlinearity
phi tuyến trong recurrence (đó là lý do vẫn scan song song được).

**Bước 2 — hệ số suy giảm phụ thuộc input, eq. (11):**

```python
log_sigmoid_lambda = F.logsigmoid(self.Lambda)          # luôn ≤ 0
a_t = torch.exp(self.c * log_sigmoid_lambda * r_t)       # eq. (11), c = 8
```

Vì `log_sigmoid_lambda ≤ 0` và `r_t ∈ (0,1)`, số mũ `c·log_sigmoid_lambda·r_t ≤ 0`, nên
`a_t ∈ (0, 1]` luôn — **đảm bảo ổn định recurrence tuyệt đối, không cần clip gì thêm**.
Khi `r_t → 1` (cổng "mở hết"), `a_t = sigmoid(Lambda)^c`; đây chính là lý do
`Lambda` được khởi tạo bằng công thức ngược (`logit(a^(1/c))`) sao cho `a` trải đều
trong `[r_min, r_max] = [0.9, 0.999]` — xem đoạn khởi tạo trong `__init__`.

Đây thực chất là kỹ thuật **"selective state space" / input-dependent gating**, cùng
họ với Mamba (Gu & Dao, 2023, `arxiv.org/abs/2312.00752`) và GLA/RecBLR — khác S4/LRU
cổ điển ở chỗ hệ số suy giảm không phải hằng số học được mà là **hàm của input hiện
tại**, giúp mô hình "quên nhanh" khi gặp hành vi bất thường và "nhớ lâu" khi chuỗi ổn
định — đúng ý nghĩa "Behavior-Dependent" trong tên gọi.

**Bước 3 — cập nhật trạng thái, eq. (12):**

```python
b_t = torch.sqrt((1.0 - a_t.pow(2)).clamp_min(1e-8)) * (i_t * x)   # phần "input mới"
h = parallel_scan(a_t, b_t, impl=self.scan_impl)                   # h_t = a_t*h_{t-1} + b_t
```

Hệ số `sqrt(1 - a_t²)` không phải ngẫu nhiên — đây là **chuẩn hoá phương sai kiểu
S4/S5/LRU**: nếu coi `h_t` là tổng có trọng số của nhiễu độc lập theo thời gian với hệ
số suy giảm hình học `a_t`, thì `Var(h_t) = Σ a_t^(2k) · Var(input)` hội tụ về
`Var(input)/(1-a_t²)`; nhân thêm `sqrt(1-a_t²)` vào input triệt tiêu hệ số đó, giữ
`Var(h_t) ≈ Var(input)` không phụ thuộc `a_t` — nếu thiếu bước này, khi `a_t` gần 1
(nhớ rất lâu) trạng thái ẩn sẽ nổ dần theo chiều dài chuỗi. Ý tưởng gốc: Orvieto et al.
2023, mục 3.2 (link ở trên); RecBLR áp dụng lại y hệt.

**Perturbation injection, eq. (17) — `_perturb`:**

```python
keep = torch.rand(x.shape[:2], ...)
replace = keep <= self.perturb_prob                  # Bernoulli(θ) độc lập theo (batch, time)
if mask is not None: replace = replace & mask.bool()  # không đụng vào padding
noise = torch.rand_like(x) * (high - low) + low        # U(low, high), mặc định U(0,1)
return x * (1 - replace_f) + noise * replace_f, replace
```

Perturbation áp dụng **trước** eq. (9) — tức là `r_t, i_t` và cả state update đều "nhìn
thấy" input đã bị nhiễu. Quan trọng: `perturb=True` chỉ được bật ở **một forward pass
thứ hai, riêng biệt**, dùng để tính `L_Per` (xem mục 5.1) — forward pass chính
(`L_Rec`) luôn sạch. Điều này giải thích tại sao log cho thấy **mỗi bước train với
`--use_perturbation` tốn gần gấp đôi thời gian** (2 forward pass thay vì 1) — đúng như
`reproduction_report.md` mục 9.2 đo được (~3.8s/iter so với ~1.9s/iter).

**Hai thuật toán scan** (`sequential_scan`, `hillis_steele_scan`) tính cùng một công
thức đệ quy `h_t = a_t·h_{t-1} + b_t` nhưng khác độ phức tạp:

- `sequential_scan`: vòng lặp Python thuần, O(L) bước tuần tự — **không thể song song
  hoá qua chiều thời gian**, chỉ dùng làm oracle đối chiếu trong test (`tests/test_scan.py`).
- `hillis_steele_scan`: thuật toán **prefix-sum song song kiểu Hillis–Steele**
  (Hillis & Steele, 1986) — tổng quát hoá của phép "parallel scan" kinh điển
  (Blelloch, *"Prefix Sums and Their Applications"*, CMU-CS-90-190,
  `cs.cmu.edu/~guyb/papers/Ble93.pdf`). Ý tưởng: sau vòng thứ `k`, mỗi vị trí đã "gộp"
  xong `2^k` bước liền trước; sau `⌈log2 L⌉` vòng thì gộp xong toàn chuỗi — độ phức tạp
  `O(L log L)` công việc nhưng **độ sâu (depth) chỉ `O(log L)`**, nên chạy nhanh trên
  GPU/nhiều core dù tổng công việc nhiều hơn bản tuần tự. Đây là kỹ thuật cốt lõi khiến
  các mô hình SSM/LRU hiện đại (S4, S5, Mamba, RecBLR, LRU) train nhanh gần bằng
  Transformer thay vì chậm như RNN cổ điển. Muốn hiểu sâu cơ chế "associative scan" áp
  dụng cho linear RNN, nên đọc thêm Martin & Cundy 2017
  (`arxiv.org/abs/1709.04057`, được trích dẫn ngay trong `triton_scan.py`) và blog
  Sasha Rush *"The Annotated S4"* (`srush.github.io/annotated-s4`).
- `_triton_scan` (trong `triton_scan.py`, copy nguyên văn từ RecBLR): cùng thuật toán
  Blelloch nhưng viết bằng **Triton kernel chạy trên GPU**, dùng mẹo "pack 2 số
  float32 vào 1 số uint64" (`pack64`/`unpack64`) để tận dụng `tl.associative_scan` có
  sẵn của Triton — chỉ chạy được khi có CUDA, không cần quan tâm nếu máy bạn chạy CPU
  (đúng tình trạng hiện tại theo `reproduction_report.md`).

**`CausalDepthwiseConv1d`**: convolution 1 chiều **depth-wise** (mỗi kênh có filter
riêng, `groups=d_model`) với `padding=kernel_size-1` rồi **cắt bỏ phần dư bên phải**
(`[..., :length]`) — đây là kỹ thuật "causal conv" chuẩn (WaveNet-style): đảm bảo output
tại vị trí `t` chỉ phụ thuộc `x[0..t]`, không nhìn tương lai. Input được nhân với
`mask` **trước** khi conv để phần padding trái không "rò rỉ" giá trị rác vào các vị trí
thật gần đó (do kernel size 4 → mỗi output nhìn 4 bước liền trước).

### 4.3 `model/slsrec.py` — lắp ráp toàn bộ kiến trúc

File này định nghĩa chính xác các thành phần trong Hình 1 của paper, theo đúng thứ tự
công thức (bảng mapping nằm ngay trong docstring đầu file). Đi từ dưới lên:

**`SLSRecEmbedding` (eq. 1):**
```python
pos = torch.arange(self.max_len - length, self.max_len, device=x.device)  # neo bên phải!
e = self.token(x) + self.position(pos).unsqueeze(0)
e = self.layer_norm(self.dropout(e))
return e * mask.unsqueeze(-1), mask
```
Vì padding nằm bên trái, positional embedding được **neo vào đầu bên phải** (item cuối
cùng luôn nhận `P[max_len-1]`) thay vì đánh số `0,1,2,...` từ trái — đúng mục 8.7 trong
`reproduction_report.md`. Với batch có độ dài cố định `max_len` (luôn đúng ở đây vì
dataloader luôn pad về đúng `max_len`), cách này **tương đương hệt** đánh số trái-phải
thông thường, nhưng có lợi thêm là bất biến với lượng padding.

**`CausalSelfAttention` — chính là SAM, eq. (8):**

Một block self-attention chuẩn (Vaswani et al., *"Attention Is All You Need"*, 2017,
`arxiv.org/abs/1706.03762`) nhưng **đơn giản hoá tối đa** theo đúng công thức paper:
1 head, không có output projection, không dropout mặc định. Điểm hay trong code là
cách xử lý **hàng query hoàn toàn bị pad**: nếu để nguyên, `softmax(-inf mọi cột)` sẽ ra
`NaN`. Code xử lý bằng cách "cho phép" mỗi vị trí tự attend vào chính nó
(`allowed = allowed | eye(L)`) rồi **zero-out kết quả sau đó bằng mask** — một mẹo nhỏ
nhưng cần thiết để tránh NaN lan ra toàn bộ batch trong lúc train.

**`SGLRULayer` — Self-Attention Gated LRU, eq. (2)–(7), khối trung tâm:**

```python
l1 = self.w1(x)                                    # (2)
g  = SAM(l1) + Conv1-D1(l1)                         # (3) -- "gate"
l2 = self.w2(x)                                     # (4)
h  = BD-LRU(Conv1-D2(l2))                            # (5) -- "giá trị" mang tính tuần tự
l3 = self.w3(g * h)                                  # (6) -- gate nhân element-wise với giá trị
return LayerNorm(Dropout(l3) + x)                    # (7) -- residual
```

Đây chính xác là ý tưởng **"gated linear unit" áp lên chuỗi**: một nhánh (`g`, kết hợp
attention toàn cục + conv cục bộ) học "nên tin bao nhiêu vào thông tin tuần tự ở vị trí
này", nhân (element-wise, không phải attention) với nhánh còn lại (`h`, do BD-LRU sinh
ra, mang thông tin tuần tự thực sự). Cùng một pattern gating với Gated Linear Units
(Dauphin et al. 2017) và các biến thể "gated SSM" (Mamba's SiLU gate, GLA). Code cho
phép tắt từng thành phần của `g` độc lập (`use_sam_gate`, `sam_conv`) — đây chính là
cách 4 ablation A/B/C/D được hiện thực (mục 8).

**`PFFN` (eq. 13) và `PFFNBlock` (eq. 14):**
```python
PFFN(x) = W8 · (GeLU(W6 x) ⊙ W7 x)     # GLU-style feed-forward, không bias
```
Giống hệt cấu trúc "SwiGLU/GEGLU feed-forward" phổ biến trong LLM hiện đại (dùng gating
nhân thay vì chỉ 1 nhánh phi tuyến như FFN Transformer gốc) — về bản chất là một GLU
khác áp trên chiều feature thay vì chiều thời gian.

**`UncertaintyEstimator` (eq. 20)** — nhánh phụ, **độc lập hoàn toàn** với nhánh chính:
có bảng embedding item/position **riêng** (`Mc`, `Pc`, không share với embedding
chính!), chạy qua 1 PFFN, rồi lấy tích vô hướng với embedding của **item nhãn đúng**
(`labels`) để ra một số vô hướng `log(σ²)` mỗi vị trí:
```python
z = self.pffn_block(xc)                # LN(Dropout(PFFN(xc) + xc)), theo eq. (20)
mc = self.embedding.token(labels)      # Mc[i, :] -- baked vào bằng ground-truth label!
return (mc * z).sum(-1)                # eq. (20): Mc[i,:]^T · LayerNorm(...)
```
Nhánh này dùng `UncertaintyPFFNBlock`: cộng residual **trước** dropout để khớp
công thức (20) được in. Backbone dùng `PFFNBlock`, cộng residual **sau** dropout
theo sơ đồ Fig. 1. Fig. 2 vẽ thứ tự khác với công thức (20); hiện thực nhánh
uncertainty chọn theo công thức.
Lưu ý quan trọng: `log(σ²)` được tính bằng cách **dùng chính nhãn đúng** (`labels`) —
điều này *chỉ hợp lệ khi training* (vì eval không có label để "gian lận"); ở
`calculate_metrics` trong `trainer/slsrec.py`, nhánh uncertainty **không hề được gọi**
khi đánh giá — model chỉ dùng `logits_from_hidden` như bình thường.

**`SLSRecModel.forward` / `logits_from_hidden` (eq. 15):**
```python
scores = x @ self.embedding.token.weight.T     # weight tying: dùng lại embedding item làm ma trận M
```
Đây là kỹ thuật **weight tying** kinh điển (Press & Wolf 2017) — ma trận dự đoán `M`
trong eq. (15) chính là ma trận embedding item transpose, không phải một `Linear` riêng
— giảm số tham số đáng kể (đây cũng là lý do bảng tham số trong
`reproduction_report.md` mục 6 cho thấy phần "Embeddings" chiếm tỉ trọng lớn: nó vừa là
input-embedding vừa là output-projection).

### 4.4 Tổng hợp: một vòng forward hoàn chỉnh trông như thế nào

```
seqs [B, L] (id item, 0 = pad, left-padded)
  -> SLSRecEmbedding: + position (neo phải) -> LayerNorm+Dropout -> x0  [B,L,D]
  -> lặp L_blocks lần (mặc định 2):
       SGLRULayer:  x -> w1 -> {SAM, Conv1-D1} -> g
                    x -> w2 -> Conv1-D2 -> BD-LRU(scan song song) -> h
                    (g ⊙ h) -> w3 -> +residual -> LayerNorm -> O
       PFFNBlock:   O -> GLU-FFN -> +residual -> LayerNorm -> x_next
  -> logits = x_L @ EmbeddingTable^T           [B, L, |I|+1]
```

Nếu `perturb=True`, **toàn bộ pipeline trên chạy lại lần 2** với input của mỗi `BDLRU`
(ở mỗi trong 2 layer, độc lập) bị thay ngẫu nhiên bởi nhiễu `U(0,1)` với xác suất `θ`
trước khi vào eq. (9).

---

## 5. Tầng loss & huấn luyện — `trainer/`

### 5.1 `trainer/slsrec.py` — nơi 4 số hạng loss (16)/(18)/(22)/(23) gặp nhau

```python
logits   = model(seqs)[0]                                  # forward "sạch"
l_rec    = CE(logits, labels)                               # eq. (16)

if use_perturbation and training:
    logits_p = model(seqs, perturb=True)[0]                 # forward THỨ HAI, có nhiễu
    l_per    = CE(logits_p, labels)                          # eq. (18)

if use_uncertainty:
    logvar   = model.log_variance(seqs, labels)              # eq. (20), log(sigma^2)
    inv_var  = exp(-logvar)                                  # 1/sigma^2
    l_unc    = inv_var * l_rec_per_position                  # eq. (22), dạng approx (beta=0)

l_total = l_rec + perturbation_weight * l_per + uncertainty_alpha * l_unc   # eq. (23)
```

Vài chi tiết tinh tế đáng chú ý:

- Toàn bộ tính theo **`reduction='none'`** rồi `_masked_mean` thủ công (bỏ qua vị trí
  `labels==0`) — vì loss phải tính **theo từng vị trí** (do convention per-position ở
  mục 3.2), không thể dùng `CrossEntropyLoss(reduction='mean')` mặc định (nó sẽ tính
  trung bình luôn cả các vị trí padding nếu không cẩn thận, dù `ignore_index=0` đã xử lý
  phần lớn việc đó — code vẫn tự làm `_masked_mean` để dùng chung `valid` mask cho cả
  4 loss, đảm bảo nhất quán).
- `l_per` chỉ được tính **khi `training=True`** — lúc validate/test không có "phiên bản
  nhiễu" nào cả, khớp với `calculate_metrics` chỉ gọi `model(seqs)` (không truyền
  `perturb=True`).
- Có 2 chế độ cho `L_Unc`: `'approx'` (mặc định, đúng eq. 22, cái paper thực sự dùng để
  suy ra eq. 23/24) và `'exact'` (đúng nghĩa đen eq. 19/21: scale *logits* bằng
  `1/σ²` rồi tính CE lại) — đây là 1 trong các cờ "phơi bày sự mơ hồ của paper" đã nói ở
  mục 2.

**`calculate_metrics`** đáng chú ý ở chỗ nó tính thêm **`val_ce`** (cross-entropy trên
tập validation, không mask) song song với Recall/NDCG — comment trong code giải thích
rõ mục đích: `train L_rec - val_ce` chính là **generalization gap**, thứ mà
`scripts/kill_tests.py --probe compare` sẽ đọc lại từ `val_metrics.jsonl` để kiểm tra
xem perturbation injection có thực sự thu hẹp gap này hay không (đúng câu hỏi bạn từng
đặt ra trước đây về mối quan hệ overfitting ↔ eq. 17/18).

### 5.2 `trainer/base.py` — vòng lặp huấn luyện dùng chung

`BaseTrainer` là **abstract base class** (Python `ABCMeta`) — `LRUTrainer` và
`SLSRecTrainer` chỉ cần override `calculate_loss` và `calculate_metrics`, mọi thứ khác
(optimizer, scheduler, checkpoint, logging, early stopping) dùng chung 100%. Vài điểm
kỹ thuật:

- **Nhóm tham số cho weight decay** (`_create_optimizer`): loại trừ `bias` và
  `layer_norm` khỏi weight decay (chuẩn thực hành tốt, tránh decay các tham số không
  nên bị co về 0). `--exclude_lambda_from_decay` cho phép loại thêm `Lambda`
  (tham số recurrence của BD-LRU) — mặc định **không loại trừ**, nghĩa là `Lambda` bị
  decay như một weight thường, khác với RecBLR gốc (weight_decay=0) — chi tiết ảnh
  hưởng nêu ở mục 8.14 `reproduction_report.md`, cũng là một chỗ mơ hồ đã được biến
  thành cờ cấu hình.
- **Hai chế độ validate**: `val_strategy='iteration'` (theo số bước, mặc định — dùng
  cho ML-1M cứ mỗi 500 bước) hoặc `'epoch'` (theo số epoch, cần để tái hiện đúng lịch
  "validate mỗi 5 epoch" trong paper — cờ `--val_epochs` được thêm riêng cho mục đích
  này).
- **Early stopping** nằm trong `BestModelLogger.log`: đếm `patience_counter`, tăng dần
  mỗi lần validate không cải thiện `best_metric` (Recall@10); `LoggerService.log_val`
  trả `True` khi `patience_counter >= early_stopping_patience` → vòng `train()` dừng.
- `train_losses.jsonl` và `val_metrics.jsonl` được ghi liên tục dạng JSON-lines — đây
  chính là 2 file mà `scripts/kill_tests.py --probe gap` và `--probe compare` đọc lại
  để phân tích **sau khi** đã train xong, không cần load lại model.

### 5.3 `trainer/lru.py`, `trainer/loggers.py`, `trainer/utils.py`

- `LRUTrainer`: y hệt logic gốc LRURec — có nhánh riêng cho dataset `xlong` (sampled
  softmax, không full-sort, vì `xlong` quá lớn để full-sort).
- `loggers.py`: `LoggerService` điều phối một danh sách `MetricGraphPrinter` (ghi từng
  metric vào TensorBoard/W&B), `RecentModelLogger` (checkpoint mới nhất) và
  `BestModelLogger` (checkpoint tốt nhất theo `best_metric`, đồng thời quản lý
  patience counter cho early stopping — đặt logic early-stopping trong logger hơi lạ về
  mặt kiến trúc phần mềm nhưng đây là style gốc từ LRURec, không phải lỗi).
- `utils.py`: cài đặt **Recall@k / MRR@k / NDCG@k** thủ công bằng tensor
  (`absolute_recall_mrr_ndcg_for_ks`) — đáng chú ý là **MRR** ở đây định nghĩa trọng số
  `1/rank` áp cho *mọi hit trong top-k* rồi sum (không chỉ lấy hit đầu tiên), do đó với
  bài toán 1-ground-truth (leave-one-out, chỉ có đúng 1 item đúng) công thức này thu
  gọn về đúng nghĩa MRR chuẩn — nhưng nếu sau này bạn đổi sang multi-label eval, cần
  soát lại công thức này. `seed_everything` chỉ là 4 dòng set seed cho
  `random/numpy/torch/os.environ` — thay thế `pytorch_lightning.seed_everything`
  (comment giải thích: bớt 1 dependency nặng chỉ để dùng 1 hàm).

---

## 6. Bộ test đơn vị — `tests/` (45 test, xác nhận: đúng, không xác nhận: hay)

Phân biệt rất rõ trong `reproduction_report.md` và cả trong docstring
`scripts/kill_tests.py`: **`tests/` chỉ trả lời "code có đúng như thiết kế
không"**, không trả lời "thiết kế có hợp lý về mặt khoa học không" (đó là việc của
kill-tests, mục 7). 6 file test, ứng với 6 mối lo kỹ thuật kinh điển khi cài SSM/LRU:

| File | Lo ngại gì | Cách kiểm |
|---|---|---|
| `test_scan.py` | scan song song có tính đúng công thức đệ quy không, gradient có chảy tới mọi tham số không | so `hillis_steele_scan` với `sequential_scan` trên nhiều độ dài `L` (kể cả không phải luỹ thừa 2) |
| `test_causality.py` | model có "nhìn trộm tương lai" không (rất dễ mắc lỗi này với conv/attention) | đổi 1 item tương lai, kiểm tra state ở mọi vị trí trước đó không đổi (< 1e-6) |
| `test_padding.py` | kết quả có phụ thuộc số lượng padding trái không | cùng 1 chuỗi, pad về 3 độ dài khác nhau (10/15/24), so kết quả |
| `test_perturbation.py` | nhiễu có đúng phân phối `U(θ)` không, có đụng vào padding không, seed có tái lập được không | so tỉ lệ thay thế thực đo với `θ`, so mean/var của `U(0,1)` (0.5, 1/12) |
| `test_losses.py` | 4 số hạng loss có cộng đúng công thức (23), có tắt đúng khi flag=False không | so `L_total` với công thức tường minh, kiểm `α=0`/`w_per=0` → loss = 0 |
| `test_tiny_overfit.py` | model có đủ capacity để học (không phải bug làm loss không giảm) | overfit 8 chuỗi tổng hợp, kiểm CE giảm > 80%, train-acc > 0.9, cho **cả 4 ablation** |

Đây là những gì bạn *nên* viết test cho bất kỳ tái hiện SSM/LRU nào — nhân quả tính, bất
biến-padding, đúng công thức scan là 3 lỗi phổ biến nhất khi tự cài linear attention/SSM
mà không test kỹ.

---

## 7. `scripts/kill_tests.py` — công cụ phản biện (khác hẳn `tests/`)

Đây là phần thú vị nhất xét theo góc độ nghiên cứu, và trực tiếp liên quan đến câu hỏi
bạn từng đặt ra: **liệu perturbation injection (eq. 17/18) có thực sự chống overfitting
như paper tuyên bố, hay chỉ là một phép biến đổi loss vô hại?** Docstring đầu file nêu
rõ triết lý: *"assume reproduction đúng (tests/ lo việc đó), giờ tấn công thẳng vào
claim khoa học của model"*. 6 probe, mỗi probe là một **cặp (giả thuyết, cách bác bỏ)**
kiểu Popper (falsificationism) chứ không phải kiểu "chứng minh đúng":

- **`probe_gap`** (không cần forward pass, chỉ đọc `train_losses.jsonl`): so
  `L_Per - L_Rec`. Nếu độ lệch này nhỏ hơn hẳn mức "nhiễu tự nhiên" giữa 2 epoch liên
  tiếp của chính `L_Rec`, kết luận: **forward pass bị nhiễu tạo ra loss gần như giống
  hệt forward sạch** → `L_total ≈ 2·L_rec`, tức chỉ là **rescale learning rate hiệu
  dụng** (AdamW bất biến với việc nhân loss bởi hằng số!), không mang thêm tín hiệu học
  nào cả.
- **`probe_transparency`**: quét `θ` từ 0.1 lên 1.0 (thay **toàn bộ** vị trí thật bằng
  nhiễu, ở **mọi** layer) và đo Recall@10/CE/độ chồng lấp top-10 thay đổi bao nhiêu.
  Nếu ngay cả phá huỷ 100% input vào BD-LRU mà đầu ra gần như không đổi → kết luận
  `θ=0.1` (giá trị paper dùng) chắc chắn "không phá hỏng gì đáng kể" — vì 10% còn yếu
  hơn 100%.
- **`probe_routing`**: dùng **forward hook** để zero hẳn output của BD-LRU
  (`H := 0`), hoặc bỏ hẳn recurrence chỉ giữ lại conv (`H := Conv1-D2`), rồi đo Recall
  rớt bao nhiêu so với full model. Nếu model "sống tốt" dù mất cả nhánh BD-LRU → nhánh
  mà perturbation nhắm vào **không phải nhánh mô hình thực sự dựa vào** để dự đoán, nên
  dù eq. 17 có "phá" nhánh đó cũng chẳng ảnh hưởng gì tới decision thực sự.
- **`probe_gates`**: instrument lại `BDLRU.forward` để log riêng giá trị `i_t` (input
  gate) tại các vị trí **bị nhiễu** so với **sạch**. Nếu `i_t` tại vị trí nhiễu thấp hơn
  hẳn (mô hình "học được cách phát hiện và đóng cổng khi thấy input trông giống nhiễu
  U(0,1)") → nhiễu **chưa bao giờ thực sự đi vào trạng thái ẩn**, coi như bị model học
  cách "lách" hoàn toàn, biến perturbation thành vô hại theo một cơ chế khác với
  `probe_routing`.
- **`probe_overfit`**: bài kiểm khắc nghiệt nhất — cho model học thuộc lòng (memorize)
  8 chuỗi tổng hợp với `θ=1.0` **và** `perturbation_weight=5` (phạt rất nặng nếu dự đoán
  sai trên bản nhiễu). Nếu model **vẫn** đạt train-accuracy > 90% dù bị "tra tấn" bằng
  nhiễu cực đoan như vậy → kết luận `L_Per` **không hề áp đặt bất kỳ chi phí capacity
  nào** lên việc ghi nhớ, nên về mặt logic **không thể** là một cơ chế chống overfitting
  theo đúng nghĩa (một regularizer thực sự phải khiến việc memorize khó hơn).
- **`probe_compare`**: đọc lại các thư mục `experiments/slsrec/<exp>/` đã train xong,
  so sánh **generalization gap** (`val_CE - train_L_rec`) và Recall@10 tốt nhất giữa
  các ablation B (không perturbation) và C (có perturbation) — nếu 2 số này nằm trong
  nhiễu ngẫu nhiên theo seed thì eq. (17)/(18) "không mua được" khả năng tổng quát hoá
  nào cả.

**Quan trọng cần hiểu đúng**: các "VERDICT: FALSIFIED" in ra trong code **không phải**
là kết luận đã được xác nhận bằng thực nghiệm trong repo này — nhìn vào
`reproduction_report.md` thì tới thời điểm viết báo cáo, model mới chạy 1–4 epoch smoke
(chưa hội tụ), và không thấy log kết quả kill-test nào được lưu lại
(`smoke_ablations.json` chỉ là output của `smoke_ablations.py`, không phải
`kill_tests.py`). Nói cách khác: **kịch bản tấn công đã viết xong và chạy được, nhưng
để có câu trả lời đáng tin cậy cho "perturbation injection có thật sự hoạt động không",
bạn cần chạy `kill_tests.py --probe all` trên checkpoint đã train hội tụ thật sự**
(không phải checkpoint smoke 3-epoch) — đây là việc còn để ngỏ, phù hợp với mục 10 của
`reproduction_report.md` ("remaining steps").

---

## 8. `scripts/smoke_ablations.py`

Ngắn gọn hơn nhiều: build dataloader thật (ML-1M), lặp qua 4 dict `overrides` ứng với
4 ablation, train vài bước + eval vài batch, in bảng tham số/loss/tốc độ, ghi ra
`smoke_ablations.json`. Mục đích **thuần kỹ thuật** (không phải khoa học như
kill-tests): xác nhận rằng cả 4 cấu hình **chạy được, không NaN, số tham số đúng như kỳ
vọng** trên dữ liệu thật — bước sanity-check trước khi chạy full training tốn hàng giờ.

---

## 9. Tóm tắt các chỗ paper mơ hồ → quyết định code hoá

(Đã trình bày chi tiết ở `reproduction_report.md` mục 8 — tóm tắt lại đây theo góc nhìn
"vì sao mỗi cờ config tồn tại", vì đây là câu hỏi hay gặp khi đọc code lần đầu):

- **eq (15)/(16) chỉ định nghĩa loss ở vị trí cuối**, nhưng harness LRURec train
  per-position → chọn per-position (theo yêu cầu nhiệm vụ), việc này lan sang cả
  eq (20) (uncertainty cũng tính per-position).
- **eq (22) (approx) vs eq (19)/(21) (exact)**: paper *nói* dùng eq (19)/(21) làm định
  nghĩa nhưng *suy ra* eq (23)/(24) — công thức thực sự tối ưu — từ xấp xỉ (22). Code
  mặc định theo (22) vì đó là cái paper *thực sự* dùng để ra kết quả cuối.
- **β=0 làm `log(σ²)` chỉ có thể tăng dần** (gradient của `-α·exp(-logvar)·L_rec` luôn
  âm khi không có β kéo lại) → về lâu dài trọng số `α/σ²` trôi về 0, nhánh uncertainty
  "tắt dần chính nó". Paper thừa nhận rõ điều này trong văn bản ("L_Rec vẫn hiệu quả kể
  cả khi σ² → +∞") — và số liệu đo thực tế trong `reproduction_report.md` mục 9.3 xác
  nhận: `L_uncertainty` giảm ~17 lần trong 3 epoch trong khi `L_rec` chỉ giảm ~13%,
  đúng như dự đoán toán học. **Đây không phải bug của code, mà là hệ quả tất yếu của
  chính công thức paper chọn** — một điểm đáng đưa vào phần phản biện nếu bạn viết bài
  liên quan đến SLSRec.
- **Activation sau conv**: eq (3)/(5) paper không ghi activation, nhưng RecBLR (được
  paper trích dẫn là nguồn cho phần conv) luôn có SiLU sau conv → code tách riêng
  `sam_conv_activation` (mặc định `none`, theo nghĩa đen paper) và
  `bdlru_conv_activation` (mặc định `silu`, theo RecBLR, vì đây là conv nằm ngay trước
  BD-LRU — thứ vay mượn trực tiếp từ RecBLR).
- **Vị trí embedding dưới padding trái**: neo phải thay vì đếm trái-phải thường (mục
  4.3 ở trên).
- **Đơn/đa-head attention, có/không output projection**: paper ghi rõ 1 ma trận
  `W_Q/W_K/W_V ∈ R^{d×d}` duy nhất → 1 head, không output proj (`sam_num_heads=1`).
- **Hạt perturbation**: paper viết "thay `x_t`" ở mức 1 bước thời gian nhưng cũng nói
  "ở mỗi layer" → code lấy 1 lần tung xúc xắc Bernoulli **độc lập cho mỗi `(chuỗi, vị
  trí, layer)`**, không dùng lại cùng 1 mask giữa các layer.
- **Weight decay lên `Lambda`**: paper nói "cấu hình giống LRURec" (decay mọi thứ trừ
  bias/layernorm) nhưng RecBLR — nguồn code recurrence — dùng `wd=0` toàn bộ → mặc định
  theo nghĩa đen (decay `Lambda`), có cờ `--exclude_lambda_from_decay` để đảo lại.

Nhìn tổng thể: **triết lý xuyên suốt của toàn bộ codebase là "đọc y hệt nghĩa đen của
công thức paper là mặc định, biến mọi lựa chọn khác thành 1 cờ CLI có thể bật"** — đây
là cách làm reproduction rất kỷ luật, giúp bạn (hoặc reviewer) truy vết chính xác nguồn
gốc mọi con số, thay vì chôn giả định ngầm trong code.

---

## 10. Toán học cốt lõi cần nắm vững (kèm link học thêm)

1. **Linear recurrence ổn định + song song hoá (associative scan)** — nền tảng của
   BD-LRU, LRU, S4, S5, Mamba:
   - Orvieto et al., *"Resurrecting Recurrent Neural Networks for Long Sequences"*,
     ICML 2023 — `arxiv.org/abs/2303.06349` (nguồn trực tiếp cho tham số hoá
     `nu/theta/gamma` trong `model/lru.py`, và ý tưởng chuẩn hoá phương sai
     `sqrt(1-a²)` mà BD-LRU kế thừa).
   - Blelloch, *"Prefix Sums and Their Applications"*, 1990 —
     `cs.cmu.edu/~guyb/papers/Ble93.pdf` (thuật toán scan song song gốc,
     nền cho cả `hillis_steele_scan` lẫn kernel Triton).
   - Martin & Cundy, *"Parallelizing Linear Recurrent Neural Nets Over Sequence
     Length"*, 2017 — `arxiv.org/abs/1709.04057` (áp dụng scan song song trực tiếp cho
     RNN tuyến tính, được trích dẫn ngay trong `triton_scan.py`).
   - Gu & Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*,
     2023 — `arxiv.org/abs/2312.00752` (họ hàng gần nhất về mặt ý tưởng
     "input-dependent decay" mà eq. 11 của SLSRec dùng).
   - Bài blog trực quan: Sasha Rush, *"The Annotated S4"* —
     `srush.github.io/annotated-s4`.

2. **Self-attention** (dùng làm 1 nhánh trong "gate" của SG-LRU): Vaswani et al.,
   *"Attention Is All You Need"*, 2017 — `arxiv.org/abs/1706.03762`.

3. **Aleatoric uncertainty qua nhiệt độ softmax kiểu Gibbs** (nền cho eq. 19–24):
   Kendall & Gal, *"What Uncertainties Do We Need in Bayesian Deep Learning for
   Computer Vision?"*, NeurIPS 2017 — `arxiv.org/abs/1703.04977` — đây gần như chắc
   chắn là công trình [45] mà paper SLSRec trích dẫn cho phép xấp xỉ eq. (22) (dạng
   "heteroscedastic uncertainty weighting": chia loss cho `σ²` học được + phạt
   `log σ`). Nắm bài này sẽ hiểu ngay vì sao thiếu số hạng `β·log σ` (khi `β=0`) khiến
   mô hình có xu hướng "gian lận" bằng cách đẩy `σ²` lên vô hạn để triệt tiêu loss —
   đúng chính là vấn đề mục 9 nêu trên.

4. **Gated Linear Units / GLU-style feed-forward** (PFFN, eq. 13): Dauphin et al.,
   *"Language Modeling with Gated Convolutional Networks"*, 2017 —
   `arxiv.org/abs/1612.08083`.

5. **Causal (depth-wise) convolution**: van den Oord et al., *"WaveNet: A Generative
   Model for Raw Audio"*, 2016 — `arxiv.org/abs/1609.03499` (nguồn gốc kỹ thuật causal
   conv bằng cách pad-trái-rồi-cắt-phải mà `CausalDepthwiseConv1d` dùng).

---

## 11. Nhận định / phản biện tổng thể

- **Về mặt kỹ thuật, reproduction này được làm rất cẩn thận**: mọi giả định mơ hồ được
  lộ diện thành cờ config thay vì chôn ngầm, có bộ test riêng cho tính đúng đắn
  (`tests/`) tách bạch khỏi bộ công cụ đánh giá tính hợp lý khoa học
  (`scripts/kill_tests.py`) — đây là một chuẩn mực tốt hiếm gặp trong code reproduction
  học thuật (đa số repo "tái hiện" trên GitHub không có test nào, chưa nói tới kill-test
  theo tinh thần Popper).

- **Điểm yếu nằm ở chính công thức của paper, không phải ở code tái hiện**: phần
  `β=0` (mục 9) là một lỗ hổng lý thuyết thực sự — cơ chế "Gibbs uncertainty scaling"
  có thể tự triệt tiêu chính nó theo thời gian huấn luyện, biến thành một dạng
  "curriculum re-weighting suy giảm dần" hơn là một regularizer bền vững. Nếu bạn định
  viết critique/insight về SLSRec (đúng yêu cầu luôn cần góc nhìn phản biện với mọi
  paper), đây là điểm mạnh nhất để đào sâu — có thể đề xuất chạy lại với `β>0` (đã có
  sẵn cờ `--uncertainty_beta`) và so sánh đường cong `σ²` theo epoch.

- **Câu hỏi bạn từng đặt ra (overfitting ↔ eq. 17/18) đã có sẵn công cụ để trả lời**
  (`probe_gap`, `probe_overfit`, `probe_compare` trong `kill_tests.py`), nhưng **chưa có
  kết quả đáng tin cậy** vì mới chạy trên checkpoint smoke chưa hội tụ. Đây là việc nên
  làm tiếp theo trước khi kết luận bất cứ điều gì về việc "perturbation injection có
  thật sự chống overfitting hay không" — chạy hội tụ thật (mục 10, `reproduction_report.md`)
  rồi mới chạy `kill_tests.py --probe all` trên checkpoint đó sẽ cho câu trả lời có ý
  nghĩa.

- **Thiết kế "1 kiến trúc, nhiều cờ" thay vì "4 model riêng"** là một lựa chọn kỹ thuật
  thông minh cho mục đích ablation, nhưng cũng có nghĩa **không có ablation nào là một
  "kiến trúc tối giản" thực sự** — ví dụ ablation A (`use_sam_gate=False,
  sam_conv=False`) vẫn chạy qua nhánh `else: g = F.silu(l1)` (dòng cuối trong
  `SGLRULayer.forward`) — một gate tuyến tính đơn giản chứ không phải "không gate nào
  cả". Khi so sánh A với B/C/D, cần nhớ rằng A **vẫn có gating**, chỉ là gating không
  học từ ngữ cảnh toàn cục (không SAM) hay cục bộ (không conv) — điều này ảnh hưởng đến
  cách diễn giải kết quả ablation nếu bạn dùng chúng cho báo cáo.
