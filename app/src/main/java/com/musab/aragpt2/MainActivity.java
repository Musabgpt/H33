package com.musab.aragpt2;

import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private QwenEngine engine;

    private TextView status;
    private EditText questionBox, answerBox, correctionBox;
    private Button askButton, correctButton, teachButton, clearButton;

    private String lastQuestion = "";
    private String lastAnswer = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        status = findViewById(R.id.status);
        questionBox = findViewById(R.id.questionBox);
        answerBox = findViewById(R.id.answerBox);
        correctionBox = findViewById(R.id.correctionBox);
        askButton = findViewById(R.id.askButton);
        correctButton = findViewById(R.id.correctButton);
        teachButton = findViewById(R.id.teachButton);
        clearButton = findViewById(R.id.clearButton);

        setWorking(true, "جاري تحميل Qwen2.5-0.5B Instruct INT4…");
        executor.execute(() -> {
            try {
                engine = new QwenEngine(this);
                int count = engine.memoryCount();
                runOnUiThread(() -> setWorking(
                        false,
                        "جاهز — Qwen يعمل محليًا. عناصر ذاكرة التعلم: " + count
                ));
            } catch (Exception ex) {
                runOnUiThread(() -> {
                    setWorking(true, "تعذر تحميل Qwen: " + safeMessage(ex));
                    clearButton.setEnabled(true);
                });
            }
        });

        askButton.setOnClickListener(v -> ask());
        correctButton.setOnClickListener(v -> rememberCorrect());
        teachButton.setOnClickListener(v -> teachCorrection());
        clearButton.setOnClickListener(v -> clearFields());
    }

    private void ask() {
        String q = questionBox.getText().toString().trim();
        if (q.isEmpty() || engine == null) return;

        correctionBox.setText("");
        answerBox.setText("");
        setWorking(true, "Qwen يفكر محليًا…");

        executor.execute(() -> {
            try {
                String answer = engine.generate(q, 96);
                lastQuestion = q;
                lastAnswer = answer;

                runOnUiThread(() -> {
                    answerBox.setText(answer.isEmpty() ? "…" : answer);
                    setWorking(false,
                            "إذا الجواب صحيح اضغط ✓، وإذا خطأ اكتب التصحيح ثم اضغط علّم");
                    correctButton.setEnabled(!answer.isEmpty());
                    teachButton.setEnabled(true);
                });
            } catch (Exception ex) {
                runOnUiThread(() ->
                        setWorking(false, "خطأ أثناء التوليد: " + safeMessage(ex)));
            }
        });
    }

    private void rememberCorrect() {
        if (engine == null || lastQuestion.isEmpty() || lastAnswer.isEmpty()) return;

        setWorking(true, "يحفظ الإجابة الصحيحة في ذاكرة التعلم المحلية…");
        executor.execute(() -> {
            try {
                String msg = engine.rememberCorrect(lastQuestion, lastAnswer);
                int count = engine.memoryCount();
                runOnUiThread(() ->
                        setWorking(false, msg + " — العناصر المحفوظة: " + count));
            } catch (Exception ex) {
                runOnUiThread(() ->
                        setWorking(false, "فشل حفظ التقييم: " + safeMessage(ex)));
            }
        });
    }

    private void teachCorrection() {
        String q = questionBox.getText().toString().trim();
        String correction = correctionBox.getText().toString().trim();
        if (engine == null || q.isEmpty() || correction.isEmpty()) return;

        String wrong = answerBox.getText().toString().trim();
        setWorking(true, "يحفظ التصحيح ويحدّث سياق التعلم المحلي…");

        executor.execute(() -> {
            try {
                String msg = engine.learnCorrection(q, wrong, correction);
                lastQuestion = q;
                lastAnswer = correction;
                int count = engine.memoryCount();

                runOnUiThread(() -> {
                    answerBox.setText(correction);
                    setWorking(false, msg + " — العناصر المحفوظة: " + count);
                    correctButton.setEnabled(true);
                });
            } catch (Exception ex) {
                runOnUiThread(() ->
                        setWorking(false, "فشل التعلم من التصحيح: " + safeMessage(ex)));
            }
        });
    }

    private void clearFields() {
        questionBox.setText("");
        answerBox.setText("");
        correctionBox.setText("");
        lastQuestion = "";
        lastAnswer = "";
        status.setText("تم مسح الحقول فقط — ذاكرة التعلم لم تُحذف");
        correctButton.setEnabled(false);
        teachButton.setEnabled(false);
    }

    private void setWorking(boolean working, String message) {
        status.setText(message);
        askButton.setEnabled(!working && engine != null);
        if (working) {
            correctButton.setEnabled(false);
            teachButton.setEnabled(false);
        }
        clearButton.setEnabled(true);
    }

    private static String safeMessage(Throwable t) {
        String m = t == null ? null : t.getMessage();
        return (m == null || m.trim().isEmpty()) ? t.getClass().getSimpleName() : m;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        if (engine != null) {
            try { engine.close(); } catch (Exception ignored) {}
        }
    }
}
