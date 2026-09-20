package com.musab.aragpt2;

import java.util.Locale;

public final class HostedProviderConfig {
    public static final String PROVIDER_GEMINI = "gemini";
    public static final String DEFAULT_GEMINI_MODEL = "gemini-3.8-flash";

    public final String provider;
    public final String model;
    public final String apiKey;

    public HostedProviderConfig(String provider, String model, String apiKey) {
        this.provider = clean(provider).toLowerCase(Locale.ROOT);
        this.model = clean(model);
        this.apiKey = clean(apiKey);
    }

    public boolean isConfigured() {
        return PROVIDER_GEMINI.equals(provider)
                && !model.isEmpty()
                && !apiKey.isEmpty();
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
