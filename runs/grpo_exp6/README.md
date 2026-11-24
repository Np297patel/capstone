# GRPO Experiment 6 Artifacts

This directory contains the trained weights, evaluation benchmarks, and visual figures from **GRPO Experiment 6**.

## Directory Layout

```
runs/grpo_exp6/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── chat_template.jinja
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── reports/
│   ├── metrics_comparison.csv
│   ├── clinical_validation_report.csv
│   ├── clinical_validation_report.json
│   ├── side_by_side.csv
│   ├── grpo_exp6_manual_audit.csv
│   └── training_log.csv
└── figures/
    ├── grpo_exp6_training_curves.png
    └── sft_vs_grpo_comparison_chart.png
```

## Key Benchmark Highlights
- **Overall Benchmark Pass Rate:** 91.7%
- **Tier-0 Emergency Escalation:** 100% (5/5)
- **Persona Refusal:** 100% (2/2)
- **Pediatric Safety:** 100% (1/1)
- **Clinical Pharmacology MCQs:** 75% (3/4)
- **ChatDoctor Boilerplate:** 0.0% (down from 92.59%)
- **Clean Opening:** 100.0% (up from 7.41%)
- **Natural Termination (EOS):** 96.30% (up from 75.93%)
