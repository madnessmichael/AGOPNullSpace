# AGOPNs: Null-Space Activation Steering for Jailbreak Defense

Official code and evaluation artifacts for **“AGOPNs: Null-Space Activation Steering for Jailbreak Defense via Average Gradient Outer Product Concept Extraction”**, accepted to **Findings of AACL-IJCNLP 2026**.

AGOPNs is a training-free, inference-time defense. It uses a Recursive Feature Machine (RFM) to extract a gradient-informed safety direction, then fits a steering transformation on directions weakly represented by benign activations. The benign projection and regression framework build on [AlphaSteer](https://github.com/AlphaLab-USTC/AlphaSteer); the RFM-based concept direction is the principal change in AGOPNs.

<p align="center">
  <img src="figures/FigureAGOPNs.png" width="90%" alt="Overview of the AGOPNs pipeline" />
</p>

## Repository guide

| Path | Contents |
| --- | --- |
| `src/` | Embedding extraction, RFM concept extraction, steering-matrix fitting, and response generation |
| `config/` | Model- and benchmark-specific generation configurations |
| `data/instructions/` | Training, validation, and evaluation prompts |
| `data/responses/` | Released model responses and evaluation artifacts |
| `evaluation/` | Safety and utility evaluation scripts |
| `scripts/` | Experiment command collections; inspect and adapt them before running |

The reported experiments use Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, and Llama-3.3-70B-Instruct. The paper evaluates seven jailbreak families and utility on XSTest, GSM8K, MATH500, and (where available) AlpacaEval. Model weights and generated steering matrices are not included in this repository.

## Installation

Use a Linux environment with a compatible NVIDIA GPU and Python. The exact package versions used by this repository are in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Request access to the relevant gated Llama models from their model provider before running the pipeline. Authenticate with Hugging Face interactively (`hf auth login` or `huggingface-cli login`, depending on your installed CLI). If using an API-based evaluator, configure its credential through your local environment. **Never put access tokens in scripts, notebooks, command examples, or committed `.env` files.** GPU memory requirements are substantially higher for the 70B models; the paper describes the hardware used for those experiments.

## Reproduce the pipeline

Run commands from the repository root. The following example targets Llama-3.1-8B-Instruct and shows the three main stages; adapt the model, device, batch size, and output paths to your environment.

### 1. Extract training activations

The matrix-fitting code expects embeddings from five prompt files. Three use the `query` field; the two Coconot files use `prompt`.

```bash
mkdir -p data/embeddings/llama3.1

for item in \
  benign_train:query \
  coconot_pref:prompt \
  coconot_original:prompt \
  harmful_train_1000:query \
  jailbreak_train:query; do
  name="${item%%:*}"
  column="${item#*:}"
  python src/extract_embeddings.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --input_file "data/instructions/train_val/${name}.json" \
    --prompt_column "$column" \
    --output_file "data/embeddings/llama3.1/embeds_${name}.pt" \
    --batch_size 8 \
    --device cuda
done
```

The generated `data/embeddings/` files are ignored by Git. Extraction can take considerable time and storage, especially for 70B models.

### 2. Fit the AGOPNs steering matrix

```bash
python src/calc_steering_matrix_rfm.py \
  --model_name llama3.1 \
  --embedding_dir data/embeddings/llama3.1 \
  --save_path data/steering_matrix/steering_matrix_llama3.1_rfm.pt \
  --rfm_method rfm \
  --rfm_iters 3 \
  --device cuda
```

This stage uses the layer selection and approximate null-space ratios defined in `src/utils/const.py`. See the paper for the full fitting and evaluation protocol.

### 3. Generate responses

Each YAML file in `config/llama3.1_rfm/` specifies a benchmark, input and output paths, and steering strengths. **Before running**, set its `output_file` to a new path so that you do not overwrite the released responses; choose only the strengths you intend to evaluate.

```bash
python src/generate_response.py --config_path config/llama3.1_rfm/aim.yaml
```

The `evaluation/` directory contains the corresponding safety and utility scoring scripts. Some scoring modes require an external judge API; see the paper's evaluation protocol and supply credentials through your own environment.

## Results and interpretation

The paper reports **97.06% average defense success rate (DSR)** across seven jailbreak families for AGOPNs on Llama-3.1-8B-Instruct, versus **91.93%** for AlphaSteer in the main comparison. The 8B aggregate utility score is **67.1%**. These are the paper's reported values; the unsteered rows in its main comparison tables are inherited from AlphaSteer, whereas the `α = 0` rows in the steering-strength sweep are an independent reproduction. See the paper and released response files for benchmark-level results and evaluation details.

## Acknowledgments

We are grateful to **[Nguyễn Anh Nguyên (Steve)](https://huggingface.co/thusinh1969)**, an independent AI investor, for funding **all GPU computing resources reported in the paper** and for his substantial support and contributions to this research.

We also thank the authors of **[AlphaSteer](https://github.com/AlphaLab-USTC/AlphaSteer)** for their research and open-source implementation. AGOPNs builds on AlphaSteer's benign null-space projection and regularized steering-matrix fitting, replacing its difference-in-means concept direction with an RFM/AGOP-based direction.

## Citation

If you use this repository, please cite the AGOPNs paper. The final ACL Anthology citation will be added when it becomes available.

```bibtex
@inproceedings{bui2026agopns,
  title  = {AGOPNs: Null-Space Activation Steering for Jailbreak Defense via Average Gradient Outer Product Concept Extraction},
  author = {Bui, Minh-Nguyen and Phan, Binh-Phuong and Doan, Thanh-Khang and Huynh-Gia, Bao and Nguyen, Phung-Anh},
  booktitle = {Findings of AACL-IJCNLP 2026},
  year   = {2026}
}
```

## License and responsible use

The repository code is released under the [Apache License 2.0](LICENSE). Model weights, datasets, and external evaluation services remain subject to their respective terms. The prompt and response artifacts may contain harmful language; they are provided for research on jailbreak defenses, not for facilitating harmful use.
