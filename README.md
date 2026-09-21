# H33 DeepSeek Coder GGUF

تطبيق Android محلي يشغّل `deepseek-ai/deepseek-coder-1.3b-instruct` بصيغة GGUF Q4_K_M عبر `llama.cpp`.

## نسخة الهاتف الخفيف

- DeepSeek‑Coder 1.3B نفسه، وليس نموذجًا بديلًا.
- GGUF Q4_K_M مع memory mapping بدل حزمة ONNX الثقيلة.
- سياق 512 token وKV cache بصيغة Q8 وCPU threads عددها 2 لهواتف 4GB RAM.
- ARM64 فقط، بلا إنترنت أو Google AI أو بحث ويب أو نماذج مستضافة.
- محادثة محلية ونسخ/حفظ TXT وسجل أدوات محلية قابل للتوسعة.
- واجهة داكنة مع معالجة شريطي النظام ولوحة المفاتيح.

## البناء الكامل والتحديث الصغير

شغّل workflow: **Build H33 DeepSeek Coder GGUF APK**.

- أول تثبيت: اترك `bundle_model=true`. الناتج `H33-DeepSeek-Coder-GGUF-FULL`.
- تحديث لاحق: اختر `bundle_model=false`. الناتج `H33-DeepSeek-Coder-GGUF-UPDATE` ولا يحمل النموذج.

النموذج يُنسخ إلى مساحة التطبيق أول مرة، والتحديثات الصغيرة تعيد استخدامه. يحفظ GitHub Actions بصمة توقيع ثابتة في cache خاص بالمستودع، ولذلك يقبل Android تحديثات هذا الفرع دون إزالة التطبيق. إذا فُقد cache التوقيع أو حُذفت بيانات التطبيق، يلزم تثبيت نسخة كاملة جديدة.

## ربط أداة محلية

أي أداة تطبّق `LocalTool` ثم تسجّل عبر `ToolRegistry.register`. لا تحصل الأداة على صلاحيات تلقائيًا؛ تنفيذ Android هو الذي يحدد صلاحياتها.
