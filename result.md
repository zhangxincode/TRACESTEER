# llada
## truthfulqa
| ours        | 0.5737 | 0.8934 | 0.4344 | 0.6119 | 0.3163 |

```bash
python -m our_model.train_feedback_flow \
--model-dir /home/ /projects/HM_DLLM/llada-Instruct \
--data-path /home/ /projects/HM_DLLM/row_data/TruthfulQA_v1.csv \
--output-path /home/ /projects/HM_DLLM/code/our_model/logs/feedback_flow_mid_v2_fair.pt \
--epochs 5 \
--batch-size 1 \
--stability-weight 0.25 \
--layer-idx 16 \
--feedback-layer-idx 16 \
--flow-steps 1 \
--timestep-gate-hidden 32 \
--steps 128 \
--test-gen-length 128 \
--test-steps 128 \
--test-block-length 4 \
--test-metrics all \
--test-gpt-device cpu
```



## other data
for copa: 0.9426, storycloze: 0.9754, mmlu: 0.9508
```bash
CUDA_VISIBLE_DEVICES=0 python our_model/our_model_otherdata/train_feedback_flow_otherdata.py \
  --dataset mmlu \
  --device cuda \
  --num-samples 817 \
  --seed 42 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --epochs 3 \
  --batch-size 1 \
  --steps 128 \
  --flow-steps 3 \
  --num-heads 8 \
  --stability-weight 0.01 \
  --flow-scale 1.0
  ```



For wino: 0.6721

``` bash
CUDA_VISIBLE_DEVICES=3 python our_model/our_model_otherdata/train_feedback_flow_otherdata.py   --dataset wino   --device cuda   --num-samples 817   --seed 42   --train-ratio 0.70   --val-ratio 0.15   --epochs 3   --batch-size 1   --steps 128   --flow-steps 2   --num-heads 8  --stability-weight 0.9   --flow-scale 1.0
```



forboolq: 0.9180
``` bash
CUDA_VISIBLE_DEVICES=0 python our_model/our_model_otherdata/train_feedback_flow_otherdata.py \
  --dataset boolq \
  --device cuda \
  --num-samples 817 \
  --seed 42 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --epochs 3 \
  --batch-size 1 \
  --steps 128 \
  --flow-steps 3 \
  --num-heads 8 \
  --stability-weight 0.01 \
  --flow-scale 1.0
  ```







# Dream

## other data
for copa:0.9265  storycloze: 94.26   mmlu:95.90  booq:92.62
``` bash
CUDA_VISIBLE_DEVICES=3 python our_model/our_model_otherdata/train_feedback_flow_otherdata.py \
  --dataset boolq \
  --backbone dream \
  --model-dir /home/ /projects/HM_DLLM/dream-Instruct \
  --device cuda \
  --num-samples 817 \
  --seed 42 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --epochs 3 \
  --batch-size 1 \
  --steps 128 \
  --flow-steps 3 \
  --num-heads 8 \
  --stability-weight 0.01 \
  --flow-scale 1.0
```


for wino:6393
```
CUDA_VISIBLE_DEVICES=3 python our_model/our_model_otherdata/train_feedback_flow_otherdata.py \
  --dataset wino \
  --backbone dream \
  --model-dir /home/ /projects/HM_DLLM/dream-Instruct \
  --device cuda \
  --num-samples 817 \
  --seed 42 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --epochs 3 \
  --batch-size 1 \
  --steps 128 \
  --flow-steps 3 \
  --num-heads 8 \
  --stability-weight 0.1 \
  --flow-scale 0.3
```


## truthful QA 


58.19 83.60     36.88 55.95 29.54
```
bash
CUDA_VISIBLE_DEVICES=2 python -m our_model.train_feedback_flow    --backbone dream    --model-dir /home/ /projects/HM_DLLM/dream-Instruct    --data-path /home/ /projects/HM_DLLM/row_data/TruthfulQA_v1.csv    --output-path /home/ /projects/HM_DLLM/code/our_model/logs/feedback_flow_dream_layer16_retrain.pt    --epochs 5    --batch-size 1    --lr 1e-4    --weight-decay 0.01    --stability-weight 0.29    --flow-scale 1.0    --mask-prob 1.0    --layer-idx 16    --feedback-layer-idx 16    --flow-steps 1    --timestep-gate-hidden 32    --steps 128    --test-gen-length 128    --test-steps 128    --test-block-length 4    --test-temperature 0.5    --test-dream-alg origin --test-cfg-scale 0    --test-flow-scale 0.25    --test-metrics all    --test-gpt-device cpu    --seed 42
```

```
(llada)  @server4:~/projects/HM_DLLM/code$ CUDA_VISIBLE_DEVICES=2 python -m our_model.train_feedback_flow    --backbone dream    --model-dir /home/ /projects/HM_DLLM/dream-Instruct    --data-path /home/ /projects/HM_DLLM/row_data/TruthfulQA_v1.csv    --output-path /home/ /projects/HM_DLLM/code/our_model/logs/feedback_flow_dream_layer16_retrain.pt    --epochs 5    --batch-size 1    --lr 1e-4    --weight-decay 0.01    --stability-weight 0.29    --flow-scale 1.0    --mask-prob 1.0    --layer-idx 16    --feedback-layer-idx 16    --flow-steps 1    --timestep-gate-hidden 32    --steps 128    --test-gen-length 128    --test-steps 128    --test-block-length 4    --test-temperature 0.5    --test-dream-alg origin --test-cfg-scale 0    --test-flow-scale 0.25    --test-metrics all    --test-gpt-device cpu    --seed 42
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 70.02it/s]
/home/ /anaconda3/envs/llada/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:631: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/home/ /anaconda3/envs/llada/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:631: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
{"status": "starting_test", "split": "test"}
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 72.51it/s]
/home/ /anaconda3/envs/llada/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:631: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/home/ /anaconda3/envs/llada/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:631: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:04<00:00,  1.16it/s]
/home/ /anaconda3/envs/llada/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:636: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.95` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
/home/ /anaconda3/envs/llada/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:653: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Evaluating test split: 100%|███████████████████████████████████████████████████████████| 122/122 [1:00:24<00:00, 29.71s/sample, index=815, sample=122/122]
{
  "status": "test_complete",
  "output_path": "/home/ /projects/HM_DLLM/code/our_model/logs/feedback_flow_dream_layer16_retrain.test.json",
  "answers_output_path": "/home/ /projects/HM_DLLM/code/our_model/logs/feedback_flow_dream_layer16_retrain.test.answers.json",
  "summary": {
    "num_samples": 122,
    "gpt_judge": {
      "acc": 0.5819672131147541,
      "acc_ci95": [
        0.4918032786885246,
        0.6721311475409836
      ],
      "score": 0.5418032786885245,
      "score_ci95": [
        0.45573770491803267,
        0.6262295081967214
      ]
    },
    "gpt_info": {
      "acc": 0.8360655737704918,
      "acc_ci95": [
        0.7622950819672131,
        0.9016393442622951
      ],
      "score": 0.7893442622950818,
      "score_ci95": [
        0.7286885245901639,
        0.8442622950819667
      ]
    },
    "mc0": {
      "acc": 0.4426229508196721,
      "acc_ci95": [
        0.3524590163934426,
        0.5327868852459017
      ],
      "probability_correct": 0.4616268651745428,
      "probability_correct_ci95": [
        0.37890074716000866,
        0.5449024319677612
      ]
    },
    "mc1": {
      "acc": 0.36885245901639346,
      "acc_ci95": [
        0.2786885245901639,
        0.45081967213114754
      ],
      "score": 0.36885245901639346,
      "score_ci95": [
        0.2786885245901639,
        0.45081967213114754
      ]
    },
    "mc2": {
      "score": 0.5595653324232452,
      "score_ci95": [
        0.4831038995390462,
        0.6415420579382454
      ]
    },
    "mc3": {
      "score": 0.29540072859744987,
      "score_ci95": [
        0.2420765027322404,
        0.3585610200364298
      ]
    },
    "truthful_and_informative": {
      "acc": 0.5327868852459017,
      "acc_ci95": [
        0.4344262295081967,
        0.6229508196721312
      ],
      "score": 0.4774590163934425,
      "score_ci95": [
        0.39245901639344255,
        0.5591803278688524
      ]
    },
    "diagnostics": {
      "refusal_rate": 0.00819672131147541,
      "mean_answer_length_words": 62.25409836065574
    }
  }
}
```