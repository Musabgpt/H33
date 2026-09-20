package com.musab.h33bench;

import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.common.model.DownloadConditions;
import com.google.mlkit.nl.translate.TranslateLanguage;
import com.google.mlkit.nl.translate.Translation;
import com.google.mlkit.nl.translate.Translator;
import com.google.mlkit.nl.translate.TranslatorOptions;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

@RunWith(AndroidJUnit4.class)
public class MlKitTranslationBenchmarkTest {
    private static String readAll(InputStream in) throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
        return out.toString(StandardCharsets.UTF_8.name());
    }

    private static String awaitTranslate(Translator translator, String text) throws Exception {
        return Tasks.await(translator.translate(text), 45, TimeUnit.SECONDS);
    }

    private static JSONObject translateVariant(
            Translator arEn, Translator enAr, JSONObject src, int goldIndex) throws Exception {
        JSONObject out = new JSONObject();
        String q = src.getString("question");
        String qEn = awaitTranslate(arEn, q);
        String qBack = awaitTranslate(enAr, qEn);
        out.put("question_en", qEn);
        out.put("question_back_ar", qBack);

        JSONArray srcChoices = src.getJSONArray("choices");
        JSONArray choicesEn = new JSONArray();
        for (int i = 0; i < srcChoices.length(); i++) {
            choicesEn.put(awaitTranslate(arEn, srcChoices.getString(i)));
        }
        out.put("choices_en", choicesEn);
        String correctBack = awaitTranslate(enAr, choicesEn.getString(goldIndex));
        out.put("correct_choice_back_ar", correctBack);
        return out;
    }

    @Test
    public void translateBenchmarkCorpus() throws Exception {
        Context target = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();

        String raw;
        try (InputStream in = testContext.getAssets().open("benchmark_input.json")) {
            raw = readAll(in);
        }
        JSONObject input = new JSONObject(raw);

        TranslatorOptions arEnOpts = new TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.ARABIC)
                .setTargetLanguage(TranslateLanguage.ENGLISH)
                .build();
        TranslatorOptions enArOpts = new TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.ENGLISH)
                .setTargetLanguage(TranslateLanguage.ARABIC)
                .build();

        Translator arEn = Translation.getClient(arEnOpts);
        Translator enAr = Translation.getClient(enArOpts);
        DownloadConditions conditions = new DownloadConditions.Builder().build();

        try {
            Tasks.await(arEn.downloadModelIfNeeded(conditions), 180, TimeUnit.SECONDS);
            Tasks.await(enAr.downloadModelIfNeeded(conditions), 180, TimeUnit.SECONDS);

            JSONArray rowsOut = new JSONArray();
            JSONArray rows = input.getJSONArray("rows");
            for (int i = 0; i < rows.length(); i++) {
                JSONObject src = rows.getJSONObject(i);
                JSONObject dst = new JSONObject();
                dst.put("domain", src.getString("domain"));
                dst.put("qid", src.getInt("qid"));
                try {
                    int gold = src.getInt("answer");
                    dst.put("msa", translateVariant(arEn, enAr, src.getJSONObject("msa"), gold));
                    dst.put("syr", translateVariant(arEn, enAr, src.getJSONObject("syr"), gold));
                    dst.put("ok", true);
                } catch (Exception ex) {
                    dst.put("ok", false);
                    dst.put("error", ex.getClass().getSimpleName() + ": " +
                            (ex.getMessage() == null ? "" : ex.getMessage()));
                }
                rowsOut.put(dst);
            }

            JSONArray probesOut = new JSONArray();
            JSONArray probes = input.getJSONArray("probes");
            for (int i = 0; i < probes.length(); i++) {
                JSONObject p = probes.getJSONObject(i);
                JSONObject o = new JSONObject();
                o.put("id", p.getString("id"));
                o.put("source", p.getString("text"));
                try {
                    String en = awaitTranslate(arEn, p.getString("text"));
                    String back = awaitTranslate(enAr, en);
                    o.put("en", en);
                    o.put("back_ar", back);
                    o.put("ok", true);
                } catch (Exception ex) {
                    o.put("ok", false);
                    o.put("error", ex.toString());
                }
                probesOut.put(o);
            }

            JSONObject result = new JSONObject();
            result.put("rows", rowsOut);
            result.put("probes", probesOut);

            File file = new File(target.getFilesDir(), "mlkit_translation_results.json");
            try (FileOutputStream out = new FileOutputStream(file)) {
                out.write(result.toString(2).getBytes(StandardCharsets.UTF_8));
                out.flush();
            }
            assertTrue(file.exists() && file.length() > 100);
        } finally {
            arEn.close();
            enAr.close();
        }
    }
}
