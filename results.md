bash meditron_eval.sh $STORAGE_ROOT/apertus/huggingface/Apertus8B
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4686|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.5098|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.7600|±  |0.0191|

# 32 nodes --> batch size 4 * 8 * 32 = 1024 

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/Meditron-Apertus-8B-only-med-no-moove
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4504|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.5027|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.7500|±  |0.0194|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/Meditron-Apertus-8B-only-med-no-moove-xav-3e-5
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4361|±  |0.0077|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4941|±  |0.0140|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6660|±  |0.0211|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-miriad-new
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3189|±  |0.0072|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3951|±  |0.0137|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.4160|±  |0.0221|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-guidelines-no-miriad-new
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2972|±  |0.0071|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3912|±  |0.0137|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.4200|±  |0.0221|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-mediset-no-miriad-new
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2984|±  |0.0071|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3134|±  |0.0130|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.5640|±  |0.0222|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-ablation-no-pubmed-no-miriad-new
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3789|±  |0.0075|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4336|±  |0.0139|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6680|±  |0.0211|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-only-mediset-3-epochs/checkpoint-15
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.3737|±  |0.0075|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.4297|±  |0.0139|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.6620|±  |0.0212|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-only-mediset-3-epochs/checkpoint-30
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2651|±  |0.0068|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.2388|±  |0.0120|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.3680|±  |0.0216|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-only-mediset-3-epochs/checkpoint-45
Status: COMPLETED
Results:
|  Tasks   |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
|----------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|
|medmcqa_g |Yaml   |strict-final-answer|     0|exact_match|↑  |0.1616|±  |0.0057|
|medqa_g   |Yaml   |strict-final-answer|     0|exact_match|↑  |0.1257|±  |0.0093|
|pubmedqa_g|Yaml   |strict-final-answer|     0|exact_match|↑  |0.3160|±  |0.0208|

bash meditron_eval.sh $STORAGE_ROOT/meditron/models/meditron-apertus-8b-only-miriad-3-epochs



# 8 nodes --> batch size 4 * 8 * 8 = 256

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



bash meditron_eval.sh $STORAGE_ROOT/meditron/models/Meditron-Apertus-8b-xav-no-miriad