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
    private AraGpt2Engine engine;

    private TextView status;
    private EditText questionBox, answerBox, correctionBox;
    private Button askButton, correctButton, teachButton, clearButton;
    private String lastQuestion = "";
    private String lastAnswer = "";

    @Override protected void onCreate(Bundle savedInstanceState) {
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

        setWorking(true, "جاري تحميل AraGPT2…");
        executor.execute(() -> {
            try {
                engine = new AraGpt2Engine(this);
                runOnUiThread(() -> setWorking(false, "جاهز — النموذج والتعلم يعملان محليًا على الهاتف"));
            } catch (Exception ex) {
                runOnUiThread(() -> {
                    setWorking(true, "تعذر تحميل النموذج: " + ex.getMessage());
                    clearButton.setEnabled(true);
                });
            }
        });

        askButton.setOnClickListener(v -> ask());
        correctButton.setOnClickListener(v -> reinforceCorrect());
        teachButton.setOnClickListener(v -> teachCorrection());
        clearButton.setOnClickListener(v -> clearFields());
    }

    private void ask() {
        String q = questionBox.getText().toString().trim();
        if (q.isEmpty() || engine == null) return;
        correctionBox.setText("");
        answerBox.setText("");
        setWorking(true, "يفكر…");

        executor.execute(() -> {
            try {
                String answer = engine.generate(q, 48, 0.75f, 40);
                lastQuestion = q;
                lastAnswer = answer;
                runOnUiThread(() -> {
                    answerBox.setText(answer.isEmpty() ? "…" : answer);
                    setWorking(false, "إذا الجواب صحيح اضغط ✓ صحيحة، وإذا خطأ اكتب التصحيح ثم اضغط علّم");
                    correctButton.setEnabled(!answer.isEmpty());
                    teachButton.setEnabled(true);
                });
            } catch (Exception ex) {
                runOnUiThread(() -> setWorking(false, "خطأ أثناء التوليد: " + ex.getMessage()));
            }
        });
    }

    private void reinforceCorrect() {
        if (engine == null || lastQuestion.isEmpty() || lastAnswer.isEmpty()) return;
        setWorking(true, "يعزز الإجابة الصحيحة ويحدّث أوزان الـAdapter…");
        executor.execute(() -> {
            try {
                String msg = engine.reinforceCorrect(lastQuestion, lastAnswer);
                long changed = engine.learnedWeightCount();
                runOnUiThread(() -> setWorking(false, msg + " — أوزان متأثرة: " + changed));
            } catch (Exception ex) {
                runOnUiThread(() -> setWorking(false, "فشل التعلم: " + ex.getMessage()));
            }
        });
    }

    private void teachCorrection() {
        String correction = correctionBox.getText().toString().trim();
        String q = questionBox.getText().toString().trim();
        if (engine == null || q.isEmpty() || correction.isEmpty()) return;
        String wrong = answerBox.getText().toString().trim();
        setWorking(true, "يتعلم من التصحيح على الجهاز…");

        executor.execute(() -> {
            try {
                String msg = engine.learnCorrection(q, wrong, correction);
                lastQuestion = q;
                lastAnswer = correction;
                long changed = engine.learnedWeightCount();
                runOnUiThread(() -> {
                    answerBox.setText(correction);
                    setWorking(false, msg + " — أوزان متأثرة: " + changed);
                    correctButton.setEnabled(true);
                });
            } catch (Exception ex) {
                runOnUiThread(() -> setWorking(false, "فشل التعلم من التصحيح: " + ex.getMessage()));
            }
        });
    }

    private void clearFields() {
        questionBox.setText("");
        answerBox.setText("");
        correctionBox.setText("");
        lastQuestion = "";
        lastAnswer = "";
        status.setText("تم مسح الحقول فقط — التعلم المحفوظ لم يُحذف");
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

    @Override protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        if (engine != null) {
            try { engine.close(); } catch (Exception ignored) {}
        }
    }
}
