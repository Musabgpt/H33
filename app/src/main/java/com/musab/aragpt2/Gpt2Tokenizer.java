package com.musab.aragpt2;

import android.content.Context;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class Gpt2Tokenizer {
    private final Map<String, Integer> encoder = new HashMap<>();
    private final Map<Integer, String> decoder = new HashMap<>();
    private final Map<String, Integer> bpeRanks = new HashMap<>();
    private final Map<Integer, String> byteEncoder;
    private final Map<Integer, Integer> byteDecoder = new HashMap<>();
    private final Map<String, String> cache = new HashMap<>();

    private static final Pattern PATTERN = Pattern.compile(
            "'(?:s|t|re|ve|m|ll|d)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );

    public Gpt2Tokenizer(Context context) throws Exception {
        String vocabText = readAsset(context, "vocab.json");
        JSONObject vocab = new JSONObject(vocabText);
        Iterator<String> vocabKeys = vocab.keys();
        while (vocabKeys.hasNext()) {
            String key = vocabKeys.next();
            int id = vocab.getInt(key);
            encoder.put(key, id);
            decoder.put(id, key);
        }

        String merges = readAsset(context, "merges.txt");
        int rank = 0;
        for (String line : merges.split("\\R")) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] p = line.split(" ");
            if (p.length >= 2) bpeRanks.put(p[0] + "\u0001" + p[1], rank++);
        }

        byteEncoder = bytesToUnicode();
        for (Map.Entry<Integer, String> e : byteEncoder.entrySet()) {
            int cp = e.getValue().codePointAt(0);
            byteDecoder.put(cp, e.getKey());
        }
    }

    public List<Integer> encode(String text) {
        ArrayList<Integer> ids = new ArrayList<>();
        Matcher m = PATTERN.matcher(text);
        while (m.find()) {
            String token = m.group();
            byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
            StringBuilder transformed = new StringBuilder();
            for (byte b : bytes) transformed.append(byteEncoder.get(b & 0xFF));
            String[] bpeTokens = bpe(transformed.toString()).split(" ");
            for (String bt : bpeTokens) {
                Integer id = encoder.get(bt);
                if (id != null) ids.add(id);
            }
        }
        return ids;
    }

    public String decode(List<Integer> ids) {
        StringBuilder merged = new StringBuilder();
        for (int id : ids) {
            String s = decoder.get(id);
            if (s != null) merged.append(s);
        }
        ArrayList<Byte> out = new ArrayList<>();
        for (int i = 0; i < merged.length();) {
            int cp = merged.codePointAt(i);
            Integer b = byteDecoder.get(cp);
            if (b != null) out.add((byte)(b & 0xFF));
            i += Character.charCount(cp);
        }
        byte[] arr = new byte[out.size()];
        for (int i = 0; i < out.size(); i++) arr[i] = out.get(i);
        return new String(arr, StandardCharsets.UTF_8);
    }

    private String bpe(String token) {
        String cached = cache.get(token);
        if (cached != null) return cached;
        List<String> word = splitCodePoints(token);
        if (word.size() <= 1) return token;

        while (true) {
            Set<String> pairs = pairs(word);
            if (pairs.isEmpty()) break;
            String best = null;
            int bestRank = Integer.MAX_VALUE;
            for (String pair : pairs) {
                Integer r = bpeRanks.get(pair);
                if (r != null && r < bestRank) {
                    bestRank = r;
                    best = pair;
                }
            }
            if (best == null) break;
            int separator = best.indexOf('\u0001');
            if (separator < 0) break;
            String first = best.substring(0, separator);
            String second = best.substring(separator + 1);
            ArrayList<String> newWord = new ArrayList<>();
            int i = 0;
            while (i < word.size()) {
                if (i < word.size() - 1 && word.get(i).equals(first) && word.get(i + 1).equals(second)) {
                    newWord.add(first + second);
                    i += 2;
                } else {
                    newWord.add(word.get(i));
                    i++;
                }
            }
            word = newWord;
            if (word.size() == 1) break;
        }
        String result = String.join(" ", word);
        if (cache.size() < 10000) cache.put(token, result);
        return result;
    }

    private static Set<String> pairs(List<String> word) {
        HashSet<String> set = new HashSet<>();
        for (int i = 0; i < word.size() - 1; i++) {
            set.add(word.get(i) + "\u0001" + word.get(i + 1));
        }
        return set;
    }

    private static List<String> splitCodePoints(String s) {
        ArrayList<String> out = new ArrayList<>();
        for (int i = 0; i < s.length();) {
            int cp = s.codePointAt(i);
            out.add(new String(Character.toChars(cp)));
            i += Character.charCount(cp);
        }
        return out;
    }

    private static Map<Integer, String> bytesToUnicode() {
        ArrayList<Integer> bs = new ArrayList<>();
        for (int i = 33; i <= 126; i++) bs.add(i);
        for (int i = 161; i <= 172; i++) bs.add(i);
        for (int i = 174; i <= 255; i++) bs.add(i);
        ArrayList<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        HashMap<Integer, String> map = new HashMap<>();
        for (int i = 0; i < bs.size(); i++) {
            map.put(bs.get(i), new String(Character.toChars(cs.get(i))));
        }
        return map;
    }

    private static String readAsset(Context context, String name) throws Exception {
        try (InputStream in = context.getAssets().open(name);
             BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = r.readLine()) != null) sb.append(line).append('\n');
            return sb.toString();
        }
    }
}
