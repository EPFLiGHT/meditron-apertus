## Experiments

### Baseline (Apertus)

bash meditron_eval.sh $STORAGE_ROOT/apertus/huggingface/Apertus8B
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4686|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.5098|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.7600|±  |0.0191|

#### High Batch size: 32 nodes --> batch size 4 * 8 * 4 * 32 = 4096:

### Main finetune (Meditron-Apertus)
bash meditron_eval.sh $STORAGE_ROOT/meditron/models/Meditron-Apertus-8B-only-med-no-moove
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4504|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.5027|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.7500|±  |0.0194|

In this eval, Meditron-Apertus is same accuracy as baseline, the number are consistent with our other evaluations, it's Apertus which got better results because I fixed the evals. This points to overfitting of Meditron-Apertus, since there is repetition in the answers. 

### Meditron-Apertus higher learning rate
bash meditron_eval.sh $STORAGE_ROOT/meditron/models/Meditron-Apertus-8B-only-med-no-moove-xav-3e-5
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4361|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4941|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6660|±  |0.0211|

Higher learning rate leads to worse performance.

### Ablations
#### No Miriad
bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-miriad-new
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3189|±  |0.0072|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3951|±  |0.0137|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.4160|±  |0.0221|

Removing Miriad from the pipeline leads to worse performance.

#### No Guidelines No Miriad
bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-guidelines-no-miriad-new
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2972|±  |0.0071|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3912|±  |0.0137|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.4200|±  |0.0221|

Removing Guidelines and Miriad is similar to No Miriad.

#### No Mediset No Miriad
bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-mediset-no-miriad-new
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2984|±  |0.0071|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3134|±  |0.0130|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.5640|±  |0.0222|

Removing Miriad and Mediset is worse than just removing Miriad.

#### No Pubmed No Miriad
bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-pubmed-no-miriad-new
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3789|±  |0.0075|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4336|±  |0.0139|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6680|±  |0.0211|

Removing Pubmed and Miriad is better than removing just Miriad -> Might point to Pubmed deteriorating MCQ performance.

#### Only Mediset
bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-only-mediset-3-epochs/checkpoint-15
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3737|±  |0.0075|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4297|±  |0.0139|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6620|±  |0.0212|

Only Mediset is better than no Miriad.

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-only-mediset-3-epochs/checkpoint-30
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2651|±  |0.0068|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2388|±  |0.0120|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.3680|±  |0.0216|

But once we go into epoch 2.

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-only-mediset-3-epochs/checkpoint-45
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.1616|±  |0.0057|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.1257|±  |0.0093|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.3160|±  |0.0208|

Or epoch 3, the model start to overfit and perfs are way worse.











# NEW

Model: /capstor/store/cscs/swissai/a127/meditron/models/Meditron-Apertus-8b-xav-no-miriad-no-warmup-5e-6
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3689|±  |0.0075|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3158|±  |0.0130|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6160|±  |0.0218|
Invalids: 1284/11912 (10.7790%)
Eval folder: eval_results/eval-apertus-8b_meditron-apertus-8b-xav-no-miriad-no-warmup-5e-6_1448409
Resps mean length: 1978.6

== eval_reports/R-eval-apertus-8b.1418629.err ==
Model: /capstor/store/cscs/swissai/a127/meditron/models/meditron-apertus-8b-only-miriad-higher-accumulation
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value|   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.432|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.491|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.724|±  |0.0200|
Invalids: 526/11912 (4.4157%)
Eval folder: eval_results/eval-apertus-8b_meditron-apertus-8b-only-miriad-higher-accumulation_1451069
Resps mean length: 3468.0

Model: /capstor/store/cscs/swissai/a127/meditron/models/Meditron-Apertus-8B-xav-trad-5e-6
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4415|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4957|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6260|±  |0.0217|
Invalids: 693/11912 (5.8177%)
Eval folder: eval_results/eval-apertus-8b_meditron-apertus-8b-xav-trad-5e-6_1448431
Resps mean length: 997.4

Model: /capstor/store/cscs/swissai/a127/meditron/models/Meditron-Apertus-8B-xav-trad-1e-6
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4733|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.5145|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.7200|±  |0.0201|
Invalids: 208/11912 (1.7461%)
Eval folder: eval_results/eval-apertus-8b_meditron-apertus-8b-xav-trad-1e-6_1451073
Resps mean length: 882.9

Model: /capstor/store/cscs/swissai/a127/meditron/models/Meditron-Apertus-8B-xav-trad-5e-6-only-miriad
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4600|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4658|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.7200|±  |0.0201|
Invalids: 421/11912 (3.5343%)
Eval folder: eval_results/eval-apertus-8b_meditron-apertus-8b-xav-trad-5e-6-only-miriad_1451072
Resps mean length: 615.7

Model: /capstor/store/cscs/swissai/a127/meditron/models/Meditron-Apertus-8B-xav-trad-5e-6-no-cont
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4406|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4831|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6340|±  |0.0216|
Invalids: 692/11912 (5.8093%)
Eval folder: eval_results/eval-apertus-8b_meditron-apertus-8b-xav-trad-5e-6-no-cont_1451071
Resps mean length: 1079.5

#### Planned next experiments:
##### Lower learning rate (explore overfitting hypothesis)
##### Remove continued pretraining (Pubmed + Guildelines)
##### Include medical reasoning dataset


















### Lower batch size (256 VS 1024) Leads to highly deteriorated performance for the ablations.
#### 8 nodes --> batch size 4 * 8 * 8 * 4 = 1024:
bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-miriad
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2333|±  |0.0065|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2726|±  |0.0125|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.3980|±  |0.0219|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-guidelines-no-miriad
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2376|±  |0.0066|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2993|±  |0.0128|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.4480|±  |0.0223|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-mediset-no-miriad
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2541|±  |0.0067|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2584|±  |0.0123|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.3700|±  |0.0216|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-pubmed-no-miriad
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.1652|±  |0.0057|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2003|±  |0.0112|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.1420|±  |0.0156|
