package com.musab.aragpt2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;

public final class SearchQueryPlan {
    public final String originalFull;
    public final String originalFocused;
    public final String englishFull;
    public final String englishFocused;

    SearchQueryPlan(
            String originalFull,
            String originalFocused,
            String englishFull,
            String englishFocused) {
        this.originalFull = clean(originalFull);
        this.originalFocused = clean(originalFocused);
        this.englishFull = clean(englishFull);
        this.englishFocused = clean(englishFocused);
    }

    public List<String> allQueries() {
        LinkedHashSet<String> unique = new LinkedHashSet<>();
        add(unique, originalFull);
        add(unique, originalFocused);
        add(unique, englishFull);
        add(unique, englishFocused);
        return Collections.unmodifiableList(new ArrayList<>(unique));
    }

    private static void add(LinkedHashSet<String> out, String value) {
        String clean = clean(value);
        if (!clean.isEmpty()) out.add(clean);
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
