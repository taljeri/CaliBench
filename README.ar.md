# CaliBench

[![arXiv](https://img.shields.io/badge/arXiv-2608.16829-b31b1b.svg)](https://arxiv.org/abs/2608.16829)

[English](README.md) | [العربية](README.ar.md)

المستودع البرمجي والبيانات الرسمية لورقة البحث:
**"CaliBench: Are the Stochastic Dynamics of Video World Models Physically Calibrated?"**
(كالي بنش: هل الديناميكيات العشوائية لنماذج العالم لتوليد الفيديو تمت معايرتها فيزيائياً؟).
هذا العمل البحثي مقبول للنشر في مجلة **TMLR** (Transactions on Machine Learning Research).

> [**CaliBench: Are the Stochastic Dynamics of Video World Models Physically Calibrated?**](https://arxiv.org/abs/2608.16829)  
> جوناثان صادقي (Jonathan Sadeghi)، جيني سايدنشوارز (Jenny Seidenschwarz)، جيسي ألرديس (Jesse Allardice)، سيريش سرينيفاسان (Sirish Srinivasan)، بنجامين غراهام (Benjamin Graham)، جيفري هوك (Jeffrey Hawke).

## كيفية الاستشهاد (How to Cite)

يرجى الاستشهاد بالورقة البحثية إذا استفدت من هذا البحث أو المستودع البرمجي في عملك:

```bibtex
@article{
sadeghi2026calibench,
title={CaliBench: Are the Stochastic Dynamics of Video World Models Physically Calibrated?},
author={Jonathan Sadeghi and Jenny Seidenschwarz and Jesse Allardice and Sirish Srinivasan and Benjamin Graham and Jeffrey Hawke},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=rWE29rkvDz},
note={}
}
}
```

**معيار قياس المعايرة لنماذج العالم لتوليد الفيديو من الصور (Calibration benchmark for image-to-video world models).**
يقيس CaliBench ما إذا كان نموذج توليد الفيديو يعيد إنتاج *التوزيع الإحصائي والفيزيائي الصحيح* للنتائج الواقعية — وليس مجرد توليد حركة تبدو مقنعة أو معقولة ظاهرياً. تتضمن المعايرة تسعة مشاهد احتمالية وعشوائية (رمي النرد، لوحة غالتون Galton board، بندول يُترك ليتأرجح يميناً أو يساراً، عجلة الروليت، سحب أوراق اللعب، كرات اليانصيب، حركة السير، مسار الكرة، وغيرها)، حيث يُشترط التوليد انطلاقاً من إطار مفرد (image-conditioned) له توزيع نتائج مرجعي حقيقي معروف تحليلياً (ground-truth outcome distribution). ويُقيّم النموذج بمقدار انحراف توزيع النتائج عبر أجيال التوليد المتعددة عن ذلك التوزيع المرجعي، مجمّعاً في مؤشر قياسي رئيسي واحد يُعرف بـ **mnTV** (المتوسط المعياري لمسافة التباين الكلي المعدلة — كلما قلت القيمة كان الأداء أفضل وأقرب للمعايرة الفيزيائية التامة).

يمثل هذا المستودع الملحق البرمجي والبيانات القابلة لإعادة الإنتاج الكاملة للورقة البحثية: خط أنابيب التوليد واستخراج النتائج، والبيانات المودعة للنتائج المستخرجة لكافة النماذج، والتحليلات الإحصائية التي تُنتج جداول وأشكال الورقة.

## آلية العمل (How it works)

لكل خلية تمثل زوجاً من `(المشهد، النموذج)` أو `(scene, model)`، يقوم خط الأنابيب بما يلي:

1. **التوليد (Generation):** توليد 32 مساراً زمنياً (rollouts) مشروطاً بإطار البداية والتوجيه النصي للمشهد (`generate_rollouts.py`، عبر واجهة Replicate API).
2. **الاستخراج (Extraction):** استخراج النتيجة الرقمية المنفصلة (discrete outcome) لكل مسار توليد باستخدام نموذج رؤية ولغة (Gemini VLM)، بالاعتماد على تصويت الأغلبية بنسبة 3 أصوات (`extract_bin.py`). وتُسجل المسارات التي لا تسفر عن نتيجة واضحة مقروءة كقيمة فارغة `null`.
3. **حساب الدرجات (Scoring):** تقييم الخلية عبر: **قابلية التسجيل (scorability)** $\rho$ (نسبة المسارات ذات النتائج المقروءة) و**مسافة التباين الكلي (total-variation distance)** بين التوزيع التجريبي والتوزيع التحليلي المرجعي للمشهد، مصححة بواسطة **نطاق العدم للعينة المحدودة (finite-sample null band)**. وتجمع نتيجة كل خلية بين انعدام الوضوح والتباين الكلي الزائد؛ ويمثل **mnTV** المتوسط الحسابي عبر المشاهد التسعة، مع فترة ثقة 95% عبر التمهيد (Bootstrap 95% CI). ويصاحب ذلك اختبار جودة المطابقة مونت-كارلو كاي تربيع ($\text{Monte-Carlo } \chi^2$) وتحليل القدرة الإحصائية (Power analysis).

تُعد تعريفات المشاهد (إطار البداية + التوجيه النصي) والتوزيعات المرجعية هي المواصفات الثابتة للمعيار القياسي؛ وكل ما عدا ذلك هو عمليات قياس وحساب.

## التثبيت (Installation)

```bash
pip install -r requirements.txt
```

تتطلب مرحلتا التوليد والاستخراج استدعاء واجهات برمجة تطبيقات مدفوعة؛ قم بضبط المفاتيح قبل تشغيل تلك المراحل:

```bash
export REPLICATE_API_TOKEN=...   # لتوليد الفيديو
export GEMINI_API_KEY=...        # لاستخراج النتائج عبر VLM
```

## البدء السريع (Quick start)

تم إيداع النتائج المستخرجة مسبقاً لكل خلية `(scene, model)` داخل مجلد `analysis/` بصيغة `<scene>_<model>_bins.json`، وبالتالي **يمكن إعادة إنتاج كافة أرقام وجداول وأشكال الورقة البحثية دون الحاجة إلى إجراء أي استدعاءات لواجهات برمجة التطبيقات**:

```bash
python compute_tv_band.py      # حساب TV لكل خلية + نطاق العدم + mnTV (المقياس الرئيسي)
python compute_mntv_ci.py      # حساب فترة الثقة 95% لـ mnTV عبر التمهيد (Bootstrap)
python compute_chi2.py         # حساب اختبار كاي تربيع، والقيم الاحتمالية p-values، والتباين الكلي
python make_scatter_grid.py    # توليد شكل شبكة 3x3 للعلاقة بين قابلية التسجيل والتباين الكلي
```

لتشغيل خط الأنابيب كاملاً من البداية إلى النهاية (التوليد ← الاستخراج ← التحليل ← توليد الأشكال والجداول)، بما في ذلك دراسات الاستئصال (Ablations) الواردة في الملحق والتحقق من مرجعية البندول:

```bash
./run_pipeline.sh
```

## هيكلية المستودع (Repository structure)

**مواصفات المعيار والكود المشترك (Benchmark specification & shared code)**
- `scenes.py` — المشاهد التسعة: إطار البداية والتوجيه النصي للتوليد.
- `constants.py` — التوزيعات المرجعية للنتائج، وأعداد النتائج لكل مشهد، والثوابت المشتركة.
- `conditioning_frames/` — إطار البداية المفرد لكل مشهد من المشاهد.
- `stats_utils.py` — أدوات اختبار جودة المطابقة مونت-كارلو كاي تربيع ($\chi^2$).

**التوليد (Generation - عبر Replicate API)**
- `generate_rollouts.py` — توليد 32 مسار فيديو من صورة لكل `(scene, model)`.
- `generate_seedance720.py`، `generate_dur5s.py` — دراسات استئصال الملحق (دقة 720p؛ ومدة موحدة 5 ثوانٍ).

**الاستخراج (Extraction - عبر Gemini VLM)**
- `extract_bin.py` — استخراج النتائج بتصويت 3 أصوات؛ وكتابة ملفات `analysis/<scene>_<model>_bins.json`.
- `extract_seedance720_gemini.py`، `extract_dur5s_gemini.py`، `extract_dice_studies.py` — مستخرجات دراسات الاستئصال.
- `validate_extraction.py` — جدول التوافق الإحصائي بين استخراج النموذج والتقييم البشري.

**التحليل والمقاييس (Analysis & metric)**
- `compute_tv_band.py` — حساب التباين الكلي TV، ونطاق العدم، ومقياس **mnTV** الرئيسي.
- `compute_mntv_ci.py` — حساب فترة الثقة 95% التمهيدية لمقياس mnTV.
- `compute_chi2.py` — اختبار $\chi^2$، والقيم الاحتمالية، والتباين الكلي لكل خلية.
- `compute_mcar_sensitivity.py` — تحليل حساسية احتساب القيم المفقودة في أسوأ وأفضل الحالات (MCAR sensitivity).
- `power_chi2.py` — تحليل القدرة الإحصائية وقابلية الاكتشاف عند $N=32$.
- `analyze_rollouts.py` — قابلية التسجيل لكل خلية والتوزيعات التجريبية.
- `verify_pendulum_v3.py` — التحقق عبر المحاكاة الفيزيائية من أن إطار البندول ينتج توزيع برنولي بنسبة نصف $\text{Bernoulli}(1/2)$.

**الأشكال والجداول (Figures & tables)**
- `make_scatter_grid.py`، `make_rollout_examples.py` — الأشكال البيانية المنشورة في الورقة.
- `make_config_table.py` — جدول إعدادات التوليد لكل نموذج (الإفصاح عن المعلمات).
- `make_seedance_ablation.py`، `make_dice_ablation_figures.py`، `recompute_dur5s.py` — دراسات استئصال الملحق.

**البيانات (Data)**
- `analysis/` — ملفات النتائج المستخرجة المودعة (`*_bins.json`) بالإضافة إلى بيانات التحقق البشري ونتائج الاستئصال؛ وتكتب نصوص التحليل مخرجاتها من ملفات JSON ورسوم وتنسيقات TeX هنا.

**نموذج Cosmos-Predict3 (Cosmos3-Super)**
- `cosmos/cosmos_config.py`، `cosmos/generate_cosmos3_minimal.py` — الإعداد المستقل ومولد المسارات للنموذج الوحيد الذي لا يعمل عبر Replicate.

يقوم السكربت `run_pipeline.sh` بتشغيل المراحل المذكورة أعلاه بالترتيب، ويوضح بالتفصيل أي الخطوات تتطلب واجهات مدفوعة مقابل الخطوات التي تُعاد إنتاجها مباشرة من البيانات المودعة.

## إعادة إنتاج نتائج الورقة البحثية (Reproducing the paper's results)

يُعد `run_pipeline.sh` المشغّل الشامل لخط الأنابيب من البداية إلى النهاية. تستدعي مرحلتا **التوليد** و**الاستخراج** واجهات برمجة تطبيقات مدفوعة وتعيدان توليد ملفات `analysis/*_bins.json`؛ يمكنك تخطيهما لإعادة إنتاج كل جدول وشكل بياني مباشرة من النتائج المودعة مسبقاً عبر مرحلتي **التحليل** و**الأشكال**. تقرأ دراسات استئصال الملحق (مقارنة SeeDance بين 480p و720p، والمدة الموحدة 5 ثوانٍ، وفحص مقياس التوجيه CFG للنرد ودراسات الأوجه المستهدفة) ملفاتها المودعة الخاصة، وتُعاد توليدها فقط باستخدام سكربتات التوليد والاستخراج المحددة لها.
