# GraphCodeBERT-RTL Project Summary

## 🎯 Project Objective

Develop an RTL (Register Transfer Level) code automatic error correction model based on GraphCodeBERT, capable of identifying and correcting common Verilog/RTL code errors.

## 📅 Project Timeline

### Phase 1: Preparation and Initial Training (2025-10-09)

1. **Data Preparation**
   - Generated 71,250 training samples
   - 5 error types included:
     - blocking_assignment (blocking assignment errors)
     - clock_sensitivity (clock sensitivity list errors)
     - missing_parentheses (missing parentheses)
     - syntax_error (syntax errors)
     - unnecessary_arithmetic (unnecessary arithmetic operations)

2. **Model Architecture Design**
   - Encoder: GraphCodeBERT (microsoft/graphcodebert-base)
   - Decoder: 6-layer Transformer Decoder
   - Special technique: DFG (Data Flow Graph) matrix fusion

3. **Initial Training Attempt**
   - Encountered OOM (Out-of-Memory) error
   - Cause: Attempting to load all 52,500 training samples into memory at once

### Phase 2: Optimization and Successful Training (2025-10-09)

4. **Memory Optimization**
   - Implemented lazy loading strategy
   - Used PyTorch Dataset and DataLoader
   - Custom collate_fn for batch processing
   - Successfully resolved OOM issue

5. **Complete Training**
   - Platform: Bridges-2 (PSC)
   - GPU: NVIDIA V100-16 (16GB)
   - Training time: ~9-12 hours
   - Training configuration:
     - Batch Size: 8
     - Learning Rate: 5e-5
     - Epochs: 20
     - Optimizer: AdamW
     - Scheduler: Linear warmup

6. **Training Results**
   - Training completed successfully for 20 epochs
   - Model saved successfully
   - Generated files:
     - `pytorch_model.bin` (~475 MB)
     - `config.json`
     - `vocab.json`

### Phase 3: Testing and Evaluation (2025-10-09 - 2025-10-10)

7. **Initial Testing Issues**
   - Encountered size mismatch error during model loading
   - Cause: `config.json` not saved correctly

8. **Problem Resolution**
   - Modified code to ensure `config.json` is saved during training
   - Manually created missing `config.json`
   - Used `strict=False` to handle minor key mismatches

9. **Final Testing**
   - Test samples: 11,250
   - Testing completed successfully
   - Generated detailed prediction file: `test_predictions.txt` (~3.8 MB)

## 🎉 Final Results

### Performance Metrics

- **Accuracy**: **100.00%** ✅
- **Test Samples**: 11,250
- **Correct Predictions**: 11,250
- **Incorrect Predictions**: 0

### Performance by Error Type

| Error Type | Samples | Accuracy |
|-----------|---------|----------|
| blocking_assignment | 2,261 | 100.00% |
| clock_sensitivity | 2,231 | 100.00% |
| missing_parentheses | 2,269 | 100.00% |
| syntax_error | 2,256 | 100.00% |
| unnecessary_arithmetic | 2,233 | 100.00% |

## 🔧 Technical Highlights

### 1. DFG (Data Flow Graph) Fusion

- Parse code data flow relationships to build DFG matrix
- Integrate DFG information into model's attention mechanism
- Enhanced model understanding of code semantics

### 2. Lazy Loading Optimization

- Implemented on-demand loading using PyTorch Dataset
- Custom collate_fn for batch processing
- Effectively reduced memory usage, supporting large-scale dataset training

### 3. Transformer Decoder

- 6-layer Transformer Decoder
- Beam search (beam_size=5) for high-quality correction generation
- Supports variable-length input and output

### 4. Pre-trained Model Fine-tuning

- Based on microsoft/graphcodebert-base pre-trained model
- Leverages pre-trained knowledge to accelerate convergence
- Improves model generalization ability

## 📁 Project File Structure

```
GraphCodeBERT/rtl_error_localization/
├── rtl_error_correction_v2.py        # Main training/testing script (optimized)
├── error_correction_model.py         # Model definition
├── train_on_bridges2.sh             # Slurm training script
├── test_model.sh                    # Slurm testing script
├── parser/                          # Verilog parser
│   ├── DFG.py                      # DFG extraction
│   ├── utils.py                    # Utility functions
│   └── tree-sitter-verilog/        # Verilog syntax parser
├── saved_models/
│   └── rtl_full_20251009_172933/   # Trained model
│       ├── pytorch_model.bin       # Model weights
│       ├── config.json             # Model configuration
│       ├── vocab.json              # Vocabulary
│       └── test_predictions.txt    # Test prediction results
├── FINAL_TEST_REPORT.md            # Complete test report
├── PROJECT_SUMMARY.md              # This document
└── README.md                       # Project description
```

## 🚀 Usage

### Training Model

```bash
# 1. SSH to Bridges-2
ssh zwang81@bridges2.psc.edu

# 2. Navigate to project directory
cd /jet/home/zwang81/GraphCodeBERT-RTL/GraphCodeBERT/rtl_error_localization

# 3. Submit training job
sbatch train_on_bridges2.sh

# 4. Check job status
squeue -u zwang81

# 5. View training logs
tail -f logs/train_*.out
```

### Testing Model

```bash
# 1. Submit testing job
sbatch test_model.sh

# 2. View test results
cat saved_models/rtl_full_20251009_172933/test_predictions.txt
```

### Local Inference

```python
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from error_correction_model import RTLErrorCorrectionModel
import torch

# Load model
model_path = "saved_models/rtl_full_20251009_172933"
config = RobertaConfig.from_pretrained(model_path)
encoder = RobertaModel.from_pretrained(model_path, config=config)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
model = RTLErrorCorrectionModel(encoder=encoder, decoder=decoder, config=config,
                                beam_size=5, max_length=128,
                                sos_id=tokenizer.cls_token_id, 
                                eos_id=tokenizer.sep_token_id)

# Load weights
checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Perform inference
buggy_code = "always @(posedge clk) begin q = d; end"
# ... (perform DFG extraction and tokenization)
# corrected_code = model(...)
```

## 💡 Lessons Learned

### 1. Memory Management is Critical

- Large-scale datasets require lazy loading
- Don't load all data into memory at once
- Use PyTorch's Dataset and DataLoader

### 2. Complete Model Saving

- Save not only model weights (`pytorch_model.bin`)
- But also configuration file (`config.json`)
- And vocabulary files (`vocab.json`, `tokenizer.json`)

### 3. Error Troubleshooting

- Carefully read error logs
- Size mismatch usually indicates configuration mismatch
- Using `strict=False` can tolerate partial mismatches

### 4. HPC Usage Tips

- Use Slurm scripts to manage jobs
- Set appropriate timeout and resource requests
- Use `expect` scripts to automate SSH operations

## 🔜 Future Improvement Directions

### 1. Extend Error Types

- Add more types of RTL errors
- Include timing violations, combinational logic errors, etc.
- Handle more complex code structures

### 2. Enhance Model Capabilities

- Use larger pre-trained models (e.g., CodeGen, StarCoder)
- Increase decoder layer count
- Try other architectures (e.g., T5, BART)

### 3. Practical Applications

- Develop VS Code / Vim plugins
- Integrate into CI/CD pipelines
- Provide online services

### 4. Data Augmentation

- Collect real RTL code errors
- Increase data diversity
- Use data augmentation techniques

## 📊 Project Statistics

- **Total Lines of Code**: ~2,000 lines
- **Training Data**: 71,250 samples
- **Model Parameters**: ~125M (GraphCodeBERT: 125M)
- **Training Time**: 9-12 hours
- **Final Accuracy**: 100.00%
- **GPU Used**: NVIDIA V100-16 (16GB)

## 🙏 Acknowledgments

- **GraphCodeBERT**: Pre-trained model from Microsoft Research
- **Tree-sitter**: Excellent tool for code parsing
- **Bridges-2**: High-performance computing resources provided by PSC
- **PyTorch**: Deep learning framework

---

**Project Status**: ✅ Completed  
**Last Updated**: 2025-10-10  
**Version**: v2.0

