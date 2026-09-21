# H33 DeepSeek Coder Lite

تطبيق Android محلي يشغّل `deepseek-ai/deepseek-coder-1.3b-instruct` بصيغة INT4 عبر ONNX Runtime GenAI.

## ما بقي في التطبيق

- نموذج واحد فقط: DeepSeek-Coder 1.3B Instruct INT4.
- محادثة محلية بذاكرة قصيرة محفوظة على الجهاز.
- بث الجواب وإيقاف التوليد ونسخ/حفظ الجواب كملف TXT.
- واجهة أدوات عامة: أي أداة محلية جديدة تطبّق `LocalTool` ثم تُسجّل في `ToolRegistry`.
- أداتان مثاليتان خفيفتان: معلومات الجهاز والوقت.

لا يملك التطبيق إذن الإنترنت، ولا يحتوي Google AI أو بحث ويب أو نموذجًا مستضافًا أو AraGPT/Qwen.

## إصلاح tokenizer

DeepSeek يستخدم تعبير GPT regex يحتوي Unicode properties مثل `\p{L}` و`\p{N}`. محرك regex في ORT GenAI Android 0.15.2 لا يقبل هذا الشكل. خطوة البناء تشغّل `scripts/patch_ort_tokenizer.py` لتحويل pre-tokenizer إلى تعبير ASCII مكافئ لمسار Python/code، ثم تنشئ `Model` و`Tokenizer` وتنفذ encode حقيقي قبل بناء APK. إذا فشل الاختبار يتوقف البناء ولا يرفع APK معطوبًا.

## البناء

من GitHub Actions شغّل **Build H33 DeepSeek Coder INT4 APK**. الناتج:

`H33-DeepSeek-Coder-Lite-APK`

البناء ARM64 فقط. النموذج نفسه يشكل معظم الحجم؛ حذف المسارات القديمة يقلل كود التطبيق واعتماداته، لكن لا يمكن جعل DeepSeek-Coder 1.3B صغيرًا مثل نموذج 0.5B دون استبداله أو خفض جودة تكميمه.

## ربط أداة

نفّذ الواجهة التالية ثم سجّلها:

```java
tools.register(new LocalTool() {
    public String name() { return "my_tool"; }
    public String displayName() { return "أداتي"; }
    public String description() { return "What the tool does and its arguments."; }
    public String execute(String arguments) throws Exception { return "result"; }
});
```

النموذج يستدعي الأداة بصيغة `<tool_call ...>`، والتطبيق يعيد النتيجة إلى DeepSeek ليكتب الجواب النهائي. لا تُمنح أي أداة صلاحيات تلقائيًا؛ صلاحياتها يحدد تنفيذها في Android.
