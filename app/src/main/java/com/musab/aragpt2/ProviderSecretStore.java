package com.musab.aragpt2;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;

import java.nio.charset.StandardCharsets;
import java.security.KeyStore;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

public final class ProviderSecretStore {
    private static final String PREFS = "h33_hosted_provider_secrets";
    private static final String KEY_ALIAS = "h33_hosted_provider_key_v1";
    private static final String KEY_CIPHERTEXT = "gemini_key_ciphertext";
    private static final String KEY_IV = "gemini_key_iv";
    private static final String KEY_MODEL = "gemini_model";

    interface Backend {
        String get(String key);
        void put(String key, String value) throws Exception;
        void remove(String key) throws Exception;
    }

    interface CipherBox {
        EncryptedSecret encrypt(String plaintext) throws Exception;
        String decrypt(String ciphertext, String iv) throws Exception;
    }

    static final class EncryptedSecret {
        final String ciphertext;
        final String iv;

        EncryptedSecret(String ciphertext, String iv) {
            this.ciphertext = clean(ciphertext);
            this.iv = clean(iv);
        }
    }

    private final Backend backend;
    private final CipherBox cipherBox;

    public ProviderSecretStore(Context context) {
        if (context == null) throw new IllegalArgumentException("context required");
        Context app = context.getApplicationContext();
        if (app == null) app = context;
        SharedPreferences prefs =
                app.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        this.backend = new SharedPreferencesBackend(prefs);
        this.cipherBox = new AndroidKeystoreCipher();
    }

    ProviderSecretStore(Backend backend, CipherBox cipherBox) {
        if (backend == null) throw new IllegalArgumentException("backend required");
        if (cipherBox == null) throw new IllegalArgumentException("cipher required");
        this.backend = backend;
        this.cipherBox = cipherBox;
    }

    public synchronized void saveGemini(String apiKey, String model) throws Exception {
        String key = clean(apiKey);
        if (key.isEmpty()) throw new IllegalArgumentException("API key required");

        String selectedModel = clean(model);
        if (selectedModel.isEmpty()) {
            selectedModel = HostedProviderConfig.DEFAULT_GEMINI_MODEL;
        }

        EncryptedSecret encrypted = cipherBox.encrypt(key);
        if (encrypted.ciphertext.isEmpty() || encrypted.iv.isEmpty()) {
            throw new IllegalStateException("فشل تشفير مفتاح Gemini");
        }

        backend.put(KEY_CIPHERTEXT, encrypted.ciphertext);
        backend.put(KEY_IV, encrypted.iv);
        backend.put(KEY_MODEL, selectedModel);
    }

    public synchronized HostedProviderConfig loadGeminiConfig() throws Exception {
        String model = clean(backend.get(KEY_MODEL));
        if (model.isEmpty()) model = HostedProviderConfig.DEFAULT_GEMINI_MODEL;

        String ciphertext = clean(backend.get(KEY_CIPHERTEXT));
        String iv = clean(backend.get(KEY_IV));
        if (ciphertext.isEmpty() || iv.isEmpty()) {
            return new HostedProviderConfig(
                    HostedProviderConfig.PROVIDER_GEMINI, model, "");
        }

        String apiKey = clean(cipherBox.decrypt(ciphertext, iv));
        return new HostedProviderConfig(
                HostedProviderConfig.PROVIDER_GEMINI, model, apiKey);
    }

    public synchronized boolean hasGeminiKey() {
        return !clean(backend.get(KEY_CIPHERTEXT)).isEmpty()
                && !clean(backend.get(KEY_IV)).isEmpty();
    }

    public synchronized void clearGemini() throws Exception {
        backend.remove(KEY_CIPHERTEXT);
        backend.remove(KEY_IV);
        backend.remove(KEY_MODEL);
    }

    private static final class SharedPreferencesBackend implements Backend {
        private final SharedPreferences prefs;

        SharedPreferencesBackend(SharedPreferences prefs) {
            this.prefs = prefs;
        }

        @Override
        public String get(String key) {
            return prefs.getString(key, "");
        }

        @Override
        public void put(String key, String value) {
            boolean ok = prefs.edit().putString(key, value).commit();
            if (!ok) throw new IllegalStateException("تعذر حفظ إعداد نموذج المتصفح");
        }

        @Override
        public void remove(String key) {
            boolean ok = prefs.edit().remove(key).commit();
            if (!ok) throw new IllegalStateException("تعذر حذف إعداد نموذج المتصفح");
        }
    }

    private static final class AndroidKeystoreCipher implements CipherBox {
        @Override
        public EncryptedSecret encrypt(String plaintext) throws Exception {
            SecretKey key = getOrCreateKey();
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, key);
            byte[] encrypted = cipher.doFinal(
                    plaintext.getBytes(StandardCharsets.UTF_8));
            return new EncryptedSecret(
                    Base64.encodeToString(encrypted, Base64.NO_WRAP),
                    Base64.encodeToString(cipher.getIV(), Base64.NO_WRAP)
            );
        }

        @Override
        public String decrypt(String ciphertext, String iv) throws Exception {
            SecretKey key = getOrCreateKey();
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            byte[] ivBytes = Base64.decode(iv, Base64.NO_WRAP);
            cipher.init(
                    Cipher.DECRYPT_MODE,
                    key,
                    new GCMParameterSpec(128, ivBytes)
            );
            byte[] clear = cipher.doFinal(
                    Base64.decode(ciphertext, Base64.NO_WRAP));
            return new String(clear, StandardCharsets.UTF_8);
        }

        private SecretKey getOrCreateKey() throws Exception {
            KeyStore store = KeyStore.getInstance("AndroidKeyStore");
            store.load(null);

            if (store.containsAlias(KEY_ALIAS)) {
                SecretKey existing = (SecretKey) store.getKey(KEY_ALIAS, null);
                if (existing == null) {
                    throw new IllegalStateException("تعذر قراءة مفتاح Android Keystore");
                }
                return existing;
            }

            KeyGenerator generator = KeyGenerator.getInstance(
                    KeyProperties.KEY_ALGORITHM_AES,
                    "AndroidKeyStore"
            );
            generator.init(
                    new KeyGenParameterSpec.Builder(
                            KEY_ALIAS,
                            KeyProperties.PURPOSE_ENCRYPT
                                    | KeyProperties.PURPOSE_DECRYPT
                    )
                            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                            .setEncryptionPaddings(
                                    KeyProperties.ENCRYPTION_PADDING_NONE)
                            .setKeySize(256)
                            .build()
            );
            return generator.generateKey();
        }
    }

    private static String clean(String value) {
        return value == null ? "" : value.trim();
    }
}
